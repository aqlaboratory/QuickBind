import torch
from torch import nn
import pytorch_lightning as pl
from commons.modified_of_modules import (
    InputEmbedder, EvoformerStack, StructureModule,
    BackboneUpdate, GatedInvariantPointAttention,
    FullEvoformerStack
)
from openfold.model.structure_module import StructureModuleTransition, InvariantPointAttention
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.rigid_utils import Rigid, Rotation
from functools import partial
from openfold.model.heads import DistogramHead
from openfold.utils.loss import distogram_loss
torch.cuda.empty_cache()

class QuickBind(nn.Module):
    def __init__(
        self,
        # INPUT EMBEDDINGS #
        aa_feat, lig_atom_feat, c_emb, c_s, c_z, use_op_edge_embed, use_pairwise_dist, use_radial_basis,
        use_rel_pos, use_multimer_rel_pos, mask_off_diagonal, one_hot_adj, use_topological_distance,
        # EVOFORMER #
        c_hidden_msa_att, c_hidden_opm, c_hidden_mul, c_hidden_pair_att, c_s_out,
        no_heads_msa, no_heads_pair, no_evo_blocks, transition_n, msa_dropout,
        pair_dropout, opm_first, chunk_size,
        # STRUCTURE MODULE #
        c_hidden, no_heads, no_qk_points, no_v_points,
        num_struct_blocks, dropout_rate,
        no_transition_layers, share_ipa_weights,
        use_gated_ipa = True, communicate = False,
        sum_pool = False, mean_pool = False, att_update = True,
        # RECYCLING #
        recycle = False, recycle_iters = 1,
        # LOSS FUNCTION #
        use_aux_head=False, use_lig_aux_head=False, no_dist_bins=64, no_dist_bins_lig=42, 
        construct_frames=True,
        # GLOBAL SETTINGS #
        use_full_evo_stack=False, blackhole_init=False,
        # OUTPUT EMBEDDING #
        output_s=False
    ):
        super(QuickBind, self).__init__()
        self.inputembedder = InputEmbedder(
            aa_feat, lig_atom_feat, c_emb, c_s, c_z, use_op_edge_embed, use_pairwise_dist, use_radial_basis,
            use_rel_pos, use_multimer_rel_pos, mask_off_diagonal, one_hot_adj, use_topological_distance
        )

        # EVOFORMER #
        if no_evo_blocks > 0:
            if use_full_evo_stack:
                self.evoformer = FullEvoformerStack(
                    c_s, c_z, c_hidden_msa_att, c_hidden_opm, c_hidden_mul, c_hidden_pair_att, c_s_out,
                    no_heads_msa, no_heads_pair, no_evo_blocks, transition_n, msa_dropout,
                    pair_dropout, opm_first=opm_first
                )
            else:
                self.evoformer = EvoformerStack(
                    c_s, c_z, c_hidden_msa_att, c_hidden_opm, c_hidden_mul, c_hidden_pair_att, c_s_out,
                    no_heads_msa, no_heads_pair, no_evo_blocks, transition_n, msa_dropout,
                    pair_dropout, opm_first=opm_first
                )
        self.no_evo_blocks = no_evo_blocks
        self.chunk_size = chunk_size

        # STRUCTURE MODULE #
        self.layer_norm_s = LayerNorm(c_s_out)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_in = Linear(c_s_out, c_s_out)
        self.num_struct_blocks = num_struct_blocks
        self.share_ipa_weights = share_ipa_weights
        if share_ipa_weights:
            self.structure_module_block = StructureModule(
                    c_s_out, c_z, c_hidden, no_heads, no_qk_points, no_v_points, dropout_rate,
                    no_transition_layers, sum_pool, mean_pool, att_update, use_gated_ipa, construct_frames
            )
        else:
            if use_gated_ipa:
                self.ipa_blocks = nn.ModuleList([
                    GatedInvariantPointAttention(
                        c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points
                    ) for _ in range(num_struct_blocks)
                ])
            else:
                self.ipa_blocks = nn.ModuleList([
                    InvariantPointAttention(
                        c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points
                    ) for _ in range(num_struct_blocks)
                ])
            self.ipa_dropout = nn.Dropout(dropout_rate)
            self.layer_norm_ipa = LayerNorm(c_s)
            self.transition = StructureModuleTransition(c_s, no_transition_layers, dropout_rate)
            self.bb_update = BackboneUpdate(c_s, sum_pool, mean_pool, att_update, construct_frames)

        # RECYCLING EMBEDDINGS #
        self.recycle = recycle
        self.recycle_iters = recycle_iters
        if recycle:
            self.layer_norm_s_recycle = LayerNorm(c_s)
            self.layer_norm_z_recycle = LayerNorm(c_z)
            self.linear_z_recycle = Linear(1, c_z)

        # AUXILIARY HEADS #
        self.use_aux_head = use_aux_head
        self.use_lig_aux_head = use_lig_aux_head
        if self.use_aux_head:
            self.distogram = DistogramHead(c_z, no_dist_bins)
        if self.use_lig_aux_head:
            self.lig_distogram = DistogramHead(c_z, no_dist_bins_lig)

        self.communicate = communicate
        if self.communicate:
            self.linear_a_i = Linear(c_s, c_z)
            self.linear_b_i = Linear(c_s, c_z)
            self.linear_dist = Linear(1, c_z)

        self.construct_frames = construct_frames
        self.blackhole_init = blackhole_init
        self.pooled_update = bool(sum_pool or mean_pool or att_update)

        self.output_s = output_s
            
    def iteration(
            self, aatype, lig_atom_features, adj, s_prev, z_prev, t_prev, ri, mask, edge_mask,
            N, t_rec, C, rec_mask, lig_mask, pseudo_N, pseudo_C
    ):
        # INPUT EMBEDDINGS #
        s, z = self.inputembedder(aatype, lig_atom_features, t_prev, edge_mask, adj, ri)
        t_lig = t_prev[:, rec_mask.shape[-1]:, :]
        if self.construct_frames and not self.blackhole_init:
            rigids = Rigid.cat(
                [
                    Rigid.from_3_points(N, t_rec, C),
                    Rigid.from_3_points(pseudo_N, t_lig, pseudo_C)
                ], dim=1
            )
        else:
            rigids = Rigid.cat(
                [
                    Rigid.from_3_points(N, t_rec, C),
                    Rigid(
                        rots = Rotation.identity(
                            shape=t_lig.shape[:-1], dtype = torch.float32, device=t_lig.device, fmt="quat"
                        ), trans = t_lig
                    )
                ], dim=1
            )

        # RECYCLING EMBEDDINGS #
        if None not in [s_prev, z_prev]:
            s_prev = self.layer_norm_s_recycle(s_prev)
            pairwise_distance_prev =  (torch.cdist(t_prev, t_prev, p=2) * edge_mask).unsqueeze(-1).to(dtype=torch.float32)
            z_prev = self.linear_z_recycle(pairwise_distance_prev) + self.layer_norm_z_recycle(z_prev)
            s = s + s_prev
            z = z + z_prev

        # EVOFORMER #
        if self.no_evo_blocks > 0:
            s = s.unsqueeze(-3)
            msa_mask = mask.unsqueeze(-2)
            s, z = self.evoformer(
                    s, z,
                    msa_mask=msa_mask,
                    pair_mask=edge_mask,
                    chunk_size=self.chunk_size
            )
        if self.recycle:
            s_prev, z_prev = s, z

        if self.output_s:
            s_pre_struct = s

        # STRUCTURE MODULE #
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        s = self.linear_in(s)

        out = []
        if self.share_ipa_weights:
            blocks = [
                partial(
                    self.structure_module_block, mask=mask, rec_mask=rec_mask, lig_mask=lig_mask
                ) for _ in range(self.num_struct_blocks)
            ]
            for block in blocks:
                s, z, new_trans = block(s, z, rigids)
                if not self.pooled_update:
                    new_trans = new_trans[:, rec_mask.shape[-1]:, :]
                new_trans = new_trans * lig_mask.unsqueeze(-1)
                if self.construct_frames:
                    rigids_ligand = rigids[:, rec_mask.shape[-1]:]
                    rigids_protein = rigids[:, :rec_mask.shape[-1]]
                    rigids_ligand_updated = rigids_ligand.compose_q_update_vec(new_trans)
                    updated_rigids = Rigid.cat([rigids_protein, rigids_ligand_updated], dim=1)
                else:
                    update = torch.cat([torch.zeros_like(rigids.get_trans()[:, :rec_mask.shape[-1], :]), new_trans], dim=-2)
                    updated_rigids = Rigid(
                        rots = rigids.get_rots(),
                        trans = rigids.get_trans() + update
                    )
                rigids = updated_rigids
                out.append(updated_rigids)
                if self.construct_frames:
                    rigids = rigids.stop_rot_gradient()
        else:
            for ipa in self.ipa_blocks:
                s = s + ipa(s, z, rigids, mask)
                s = self.ipa_dropout(s)
                s = self.layer_norm_ipa(s)
                s = self.transition(s)
                new_trans = self.bb_update(s, rec_mask, lig_mask)
                if not self.pooled_update:
                    new_trans = new_trans[:, rec_mask.shape[-1]:, :]
                new_trans = new_trans * lig_mask.unsqueeze(-1)
                if self.construct_frames:
                    rigids_ligand = rigids[:, rec_mask.shape[-1]:]
                    rigids_protein = rigids[:, :rec_mask.shape[-1]]
                    rigids_ligand_updated = rigids_ligand.compose_q_update_vec(new_trans)
                    updated_rigids = Rigid.cat([rigids_protein, rigids_ligand_updated], dim=1)                      
                else:
                    update = torch.cat([torch.zeros_like(rigids.get_trans()[:, :rec_mask.shape[-1], :]), new_trans], dim=-2)
                    updated_rigids = Rigid(
                        rots = rigids.get_rots(),
                        trans = rigids.get_trans() + update
                    )
                rigids = updated_rigids
                out.append(updated_rigids)
                if self.communicate:
                    ti = rigids.get_trans()
                    a_i = self.linear_a_i(s)
                    b_i = self.linear_b_i(s)
                    pair_emb = a_i[..., None, :] + b_i[..., None, :, :]
                    dist = (torch.cdist(ti, ti, p=2) * edge_mask).unsqueeze(-1).to(dtype=torch.float32)
                    pairwise_distance = self.linear_dist(dist)
                    pair_emb = pair_emb + pairwise_distance
                    z = z + pair_emb
                if self.construct_frames:
                    rigids = rigids.stop_rot_gradient()

        if self.recycle: t_prev = rigids.get_trans()

        if (self.use_aux_head or self.use_lig_aux_head) and self.is_final_iter:
            if self.output_s:
                return out, s, s_prev, z, t_prev, s_pre_struct
            else:
                return out, s, s_prev, z, t_prev
        else:
            if self.output_s:
                return out, s, s_prev, z_prev, t_prev, s_pre_struct
            else:
                return out, s, s_prev, z_prev, t_prev

    def forward(self, aatype, lig_atom_features, adj, rec_mask, lig_mask, N, t_rec, C, t_lig, ri, pseudo_N, pseudo_C):
        is_grad_enabled = torch.is_grad_enabled()
        # RECYCLING #
        s_prev, z_prev = None, None
        t_prev = torch.cat([t_rec, t_lig], dim=-2)
        mask = torch.cat([rec_mask, lig_mask], dim=-1).to(dtype=torch.float32)
        edge_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        for iteration in range(self.recycle_iters):
            self.is_final_iter = (iteration == (self.recycle_iters-1))
            with torch.set_grad_enabled(is_grad_enabled and self.is_final_iter):
                if self.is_final_iter and torch.is_autocast_enabled(): # Sidestep AMP bug (PyTorch issue #65766)
                    torch.clear_autocast_cache()
                if self.output_s:
                    outputs, s, s_prev, z_prev, t_prev, s_pre_struct = self.iteration(
                        aatype, lig_atom_features, adj, s_prev, z_prev, t_prev, ri, mask, edge_mask,
                        N, t_rec, C, rec_mask, lig_mask, pseudo_N, pseudo_C
                    )
                else:
                    outputs, s, s_prev, z_prev, t_prev = self.iteration(
                        aatype, lig_atom_features, adj, s_prev, z_prev, t_prev, ri, mask, edge_mask,
                        N, t_rec, C, rec_mask, lig_mask, pseudo_N, pseudo_C
                    )
                if not self.is_final_iter: del outputs, s

        if self.use_aux_head and self.use_lig_aux_head:
            distogram_logits_full = self.distogram(z_prev)
            distogram_logits_lig = self.lig_distogram(z_prev[:, rec_mask.shape[-1]:, rec_mask.shape[-1]:])
            distogram_logits = (distogram_logits_full, distogram_logits_lig)
            if self.output_s:
                return outputs, distogram_logits, s_pre_struct
            else:
                return outputs, distogram_logits
        elif self.use_aux_head:
            distogram_logits = self.distogram(z_prev)
            if self.output_s:
                return outputs, distogram_logits, s_pre_struct
            else:
                return outputs, distogram_logits
        elif self.use_lig_aux_head:
            distogram_logits = self.lig_distogram(z_prev[:, rec_mask.shape[-1]:, rec_mask.shape[-1]:])
            if self.output_s:
                return outputs, distogram_logits, s_pre_struct
            else:
                return outputs, distogram_logits
        else:
            if self.output_s:
                return outputs, s_pre_struct
            else:
                return outputs

class QuickBind_PL(pl.LightningModule):
    def __init__(
        self,
        # INPUT EMBEDDINGS #
        aa_feat, lig_atom_feat, c_emb, c_s, c_z, use_op_edge_embed,
        use_pairwise_dist, use_radial_basis, use_rel_pos, use_multimer_rel_pos,
        mask_off_diagonal, one_hot_adj, use_topological_distance,
        # EVOFORMER #
        c_hidden_msa_att, c_hidden_opm, c_hidden_mul, c_hidden_pair_att, c_s_out,
        no_heads_msa, no_heads_pair, no_evo_blocks, transition_n, msa_dropout,
        pair_dropout, opm_first, chunk_size,
        # STRUCTURE MODULE #
        c_hidden, no_heads, no_qk_points, no_v_points,
        num_struct_blocks, dropout_rate,
        no_transition_layers, share_ipa_weights,
        use_gated_ipa = False, communicate = False,
        sum_pool = False, mean_pool = False, att_update=False,
        # RECYCLING #
        recycle = False, recycle_iters = 1,
        # LOSS FUNCTION #
        loss_config = None,
        use_aux_head=False, use_lig_aux_head=False, no_dist_bins=64, no_dist_bins_lig=42,
        construct_frames=False,
        use_full_evo_stack=False, blackhole_init=False,
        # LEARNING RATE #
        lr=1.0e-5, weight_decay=1.0e-4,
    ):
        super().__init__()
        self.model = QuickBind(
            # INPUT EMBEDDINGS #
            aa_feat, lig_atom_feat, c_emb, c_s, c_z, use_op_edge_embed,
            use_pairwise_dist, use_radial_basis, use_rel_pos, use_multimer_rel_pos,
            mask_off_diagonal, one_hot_adj, use_topological_distance,
            # EVOFORMER #
            c_hidden_msa_att, c_hidden_opm, c_hidden_mul, c_hidden_pair_att, c_s_out,
            no_heads_msa, no_heads_pair, no_evo_blocks, transition_n, msa_dropout,
            pair_dropout, opm_first, chunk_size,
            # STRUCTURE MODULE #
            c_hidden, no_heads, no_qk_points, no_v_points,
            num_struct_blocks, dropout_rate,
            no_transition_layers, share_ipa_weights,
            use_gated_ipa, communicate,
            sum_pool, mean_pool, att_update,
            # RECYCLING #
            recycle, recycle_iters,
            # AUXILIARY HEADS #
            use_aux_head, use_lig_aux_head, no_dist_bins, no_dist_bins_lig,
            construct_frames, use_full_evo_stack, blackhole_init
        )
        
        self.loss = QuickBindLoss(**loss_config, use_aux_head=use_aux_head, use_lig_aux_head=use_lig_aux_head)
        self.use_aux_head = use_aux_head
        self.use_lig_aux_head = use_lig_aux_head
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(*batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, idx):
        batch, t_true = batch
        _, _, _, rec_mask, lig_mask, _, _, _, _, _, _, _ = batch
        if self.use_aux_head or self.use_lig_aux_head:
            outputs, distogram_logits = self.model(*batch)
        else:
            outputs = self.model(*batch)
            distogram_logits = None
        loss, (
            lig_lig_loss, lig_rec_loss, aux_loss, steric_clash_loss, full_distogram_loss
        ), rmsd = self.loss(t_true, outputs, lig_mask, rec_mask, distogram_logits)
        self.log('train_loss', loss)
        self.log('train_lig_lig_loss', lig_lig_loss)
        self.log('train_lig_rec_loss', lig_rec_loss)
        self.log('train_aux_loss', aux_loss)
        self.log('train_steric_clash_loss', steric_clash_loss)
        self.log('train_full_distogram_loss', full_distogram_loss)
        self.log('train_rmsd', rmsd)
        return loss

    def validation_step(self, batch, idx):
        batch, t_true = batch
        _, _, _, rec_mask, lig_mask, _, _, _, _, _, _, _ = batch
        if self.use_aux_head or self.use_lig_aux_head:
            outputs, distogram_logits = self.model(*batch)
        else:
            outputs = self.model(*batch)
            distogram_logits = None
        loss, (
            lig_lig_loss, lig_rec_loss, aux_loss, steric_clash_loss, full_distogram_loss
        ), rmsd = self.loss(t_true, outputs, lig_mask, rec_mask, distogram_logits,)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_lig_lig_loss', lig_lig_loss, sync_dist=True)
        self.log('val_lig_rec_loss', lig_rec_loss, sync_dist=True)
        self.log('val_aux_loss', aux_loss, sync_dist=True)
        self.log('val_steric_clash_loss', steric_clash_loss, sync_dist=True)
        self.log('val_full_distogram_loss', full_distogram_loss, sync_dist=True)
        self.log('val_rmsd', rmsd, sync_dist=True)
        return loss

class QuickBindLoss(nn.Module):
    def __init__(
        self, lig_lig_loss_weight, lig_rec_loss_weight, aux_loss_weight,
        steric_clash_loss_weight, full_distogram_loss_weight, clamp_distance = None, eps = 1e-8,
        use_aux_head=False, use_lig_aux_head=False,
    ):
        super().__init__()
        self.lig_lig_loss_weight = lig_lig_loss_weight
        self.lig_rec_loss_weight = lig_rec_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.steric_clash_loss_weight = steric_clash_loss_weight
        self.full_distogram_loss_weight = full_distogram_loss_weight
        self.eps = eps
        self.clamp_distance = clamp_distance
        self.use_aux_head = use_aux_head
        self.use_lig_aux_head = use_lig_aux_head

    def compute_fape_lig_lig(
        self,
        pred_frames: Rigid,
        target_frames: Rigid,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        # [*, N_frames, N_frames, 3]
        local_pred_pos = pred_frames.invert()[..., None].apply(
            pred_positions[..., None, :, :],
        )
        local_target_pos = target_frames.invert()[..., None].apply(
            target_positions[..., None, :, :],
        )
        error = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + self.eps
        )
        edge_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        error = error * edge_mask
        error = torch.sum(torch.sum(error, dim=-1), dim=-1) / torch.sum(mask, dim=-1)**2
        return torch.mean(error)

    def compute_fape_lig_rec(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        protein_frames: Rigid,
        lig_mask: torch.Tensor,
        rec_mask: torch.Tensor,
        clamp_distance = None, 
    ) -> torch.Tensor:
        # [*, N_protein_frames, N_lig_frames, 3]
        local_pred_pos = protein_frames.invert()[..., None].apply(
            pred_positions[..., None, :, :],
        )
        local_target_pos = protein_frames.invert()[..., None].apply(
            target_positions[..., None, :, :],
        )
        error = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + self.eps
        )
        edge_mask = rec_mask.unsqueeze(-1) * lig_mask.unsqueeze(-2)
        error = error * edge_mask
        if clamp_distance is not None:
            error = torch.clamp(error, min=0, max=clamp_distance)
        error = torch.sum(torch.sum(error, dim=-1), dim=-1) / (torch.sum(rec_mask, dim=-1) * torch.sum(lig_mask, dim=-1))
        return torch.mean(error)

    def compute_rmsd(self, ti, t_true, mask): 
        error = (ti - t_true) * mask.unsqueeze(-1)
        error = torch.sum(torch.sum(error**2, dim=-1), dim=-1) / (torch.sum(mask, dim=-1))
        return torch.mean(torch.sqrt(error + self.eps))

    def compute_steric_clash_loss_lig(self, ti, lig_mask):
        edge_mask = lig_mask.unsqueeze(-1) * lig_mask.unsqueeze(-2)
        pairwise_distances = torch.cdist(ti, ti, p=2) * edge_mask
        error = torch.nn.functional.relu(0.5 - pairwise_distances)
        error = torch.sum(torch.sum(torch.tril(error, diagonal=-1), dim=-1), dim=-1)
        return torch.mean(error)

    def compute_kabsch_rmsd(self, ti_batch, t_true_batch, mask):
        transformed_coords = []
        for ti, t_true in zip(ti_batch, t_true_batch):
            try:
                lig_coords_pred_mean = ti.mean(dim=0, keepdim=True, dtype=torch.float32)  # (1,3)
                lig_coords_mean = t_true.mean(dim=0, keepdim=True, dtype=torch.float32)  # (1,3)
                A = ((ti - lig_coords_pred_mean).transpose(0, 1) @ (t_true - lig_coords_mean)).to(dtype=torch.float32)
                U, S, Vt = torch.linalg.svd(A)
                corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=ti.device))
                rotation = (U @ corr_mat) @ Vt
                translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
                transformed_coords.append((rotation @ t_true.t()).t() + translation)
                return self.compute_pos_loss(ti_batch, torch.stack(transformed_coords), mask)
            except Exception:
                print('Computing Kabsch RMSD failed.')
                return torch.zeros(1, requires_grad=True, dtype=torch.float32, device=ti_batch.device)

    def compute_pos_loss(self, ti, t_true, mask):
        error = (ti - t_true) * mask.unsqueeze(-1)    
        error = torch.sum(torch.sum(error**2, dim=-1), dim=-1) / (3*torch.sum(mask, dim=-1))
        return torch.mean(error)       

    def forward(self, target_frames, outputs, lig_mask, rec_mask, distogram_logits):
        target_frames = target_frames.cuda()
        pred_frames = outputs[-1][:, rec_mask.shape[-1]:]
        rec_frames = outputs[-1][:, :rec_mask.shape[-1]]
        target_positions = target_frames.get_trans()
        pred_positions = pred_frames.get_trans()
        lig_lig_loss = self.compute_fape_lig_lig(pred_frames, target_frames, pred_positions, target_positions, lig_mask)
        lig_rec_loss = self.compute_fape_lig_rec(pred_positions, target_positions, rec_frames, lig_mask, rec_mask, self.clamp_distance)
        aux_loss = torch.mean(torch.stack([
            self.compute_fape_lig_rec(pred_frames[:, rec_mask.shape[-1]:].get_trans(), target_positions, rec_frames, lig_mask, rec_mask, self.clamp_distance) for pred_frames in outputs
        ]))
        steric_clash_loss = self.compute_kabsch_rmsd(pred_positions, target_positions, lig_mask) if self.steric_clash_loss_weight > 0 else 0.0
        rmsd = self.compute_rmsd(pred_positions, target_positions, lig_mask)

        if self.use_aux_head and self.use_lig_aux_head:
            distogram_logits_full, distogram_logits_lig = distogram_logits
            pseudo_beta_mask = torch.cat([rec_mask, lig_mask], dim=-1)
            pseudo_beta = torch.cat([rec_frames.get_trans(), pred_positions], dim=-2)
            rec_lig_distogram_loss = distogram_loss(distogram_logits_full, pseudo_beta, pseudo_beta_mask, min_bin=2.3125, max_bin=21.6875, no_bins=64)
            lig_lig_distogram_loss = distogram_loss(distogram_logits_lig, pred_positions, lig_mask, min_bin=1., max_bin=5., no_bins=42)
            full_distogram_loss = rec_lig_distogram_loss + lig_lig_distogram_loss
        elif self.use_aux_head:
            pseudo_beta_mask = torch.cat([rec_mask, lig_mask], dim=-1)
            pseudo_beta = torch.cat([rec_frames.get_trans(), pred_positions], dim=-2)
            full_distogram_loss = distogram_loss(distogram_logits, pseudo_beta, pseudo_beta_mask, min_bin=2.3125, max_bin=21.6875, no_bins=64)
        elif self.use_lig_aux_head:
            full_distogram_loss = distogram_loss(distogram_logits, pred_positions, lig_mask, min_bin=1., max_bin=5., no_bins=42)
        else:
            full_distogram_loss = 0.0

        loss = (
            self.lig_lig_loss_weight * lig_lig_loss + \
            self.lig_rec_loss_weight * lig_rec_loss + \
            self.aux_loss_weight * aux_loss +\
            self.steric_clash_loss_weight * steric_clash_loss +\
            self.full_distogram_loss_weight * full_distogram_loss
        )

        if torch.isnan(loss):
            print('Loss is nan, skipping...')
            loss = torch.zeros(1, requires_grad=True, dtype=torch.float32, device=lig_lig_loss.device)

        return loss, (lig_lig_loss, lig_rec_loss, aux_loss, steric_clash_loss, full_distogram_loss), rmsd
