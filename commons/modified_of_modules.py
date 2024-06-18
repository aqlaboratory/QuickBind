# This file contains, to a large extent, modified OpenFold code.
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

import torch
import math
from torch import nn
from typing import Optional, Sequence, Tuple, List
from openfold.utils.tensor_utils import permute_final_dims
from openfold.model.structure_module import (
    StructureModuleTransition, InvariantPointAttention,
    ipa_point_weights_init_, flatten_final_dims
)
from openfold.utils.chunk_utils import chunk_layer
from openfold.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
from openfold.model.evoformer import DropoutRowwise
from openfold.model.evoformer import MSATransition
from openfold.model.outer_product_mean import OuterProductMean
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from openfold.model.triangular_attention import TriangleAttention
from openfold.model.pair_transition import PairTransition
from openfold.utils.tensor_utils import add
from openfold.model.primitives import (
    Attention, Linear, LayerNorm, 
    _attention_chunked_trainable,
)
from openfold.utils.rigid_utils import Rigid
from functools import partial
import importlib

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
torch.cuda.empty_cache()

class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.
    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """
    def __init__(
        self,
        aa_feat: int,
        lig_atom_feat: int,
        c_emb: int,
        c_s: int,
        c_z: int,
        use_op_edge_embed: bool = False,
        use_pairwise_dist: bool = False,
        use_radial_basis: bool = False,
        use_rel_pos: bool = False,
        use_multimer_rel_pos: bool = False,
        mask_off_diagonal: bool = False,
        one_hot_adj: bool = False,
        use_topological_distance: bool = False,
        relpos_k: int = 32,
        **kwargs,
    ):
        """
        Args:
            aa_feat:
                Receptor feature dimension
            lig_atom_feat:
                Ligand feature dimension
            c_emb:
                Receptor and ligand feature embedding dimension
            c_s:
                Final dimension of the target features
            c_z:
                Pair embedding dimension
            use_op_edge_embed:
                Use the OuterProductMean module to create the initial pair embedding. 
            use_pairwise_dist:
                Use pairwise distances in the pair embedding.
            use_radial_basis:
                Embed the pairwise distances using radial basis functions.
            use_rel_pos:
                Use relative positional encoding and the ligand adjacency matrix.
            use_multimer_rel_pos:
                Use AF-Multimer relative positional encoding for the protein.
            mask_off_diagonal:
                Mask intermolecular pairwise distances.
            one_hot_adj:
                One-hot encode bond types in the adjacency matrix.
            use_topological_distance:
                Use the topological distance in the adjacency matrix.                
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedder, self).__init__()

        self.c_z = c_z
        self.linear_aatype = Linear(aa_feat, c_emb)
        self.linear_lig = Linear(lig_atom_feat, c_emb)
        self.linear_si = Linear(c_emb, c_s)
        self.use_op_edge_embed = use_op_edge_embed
        if use_op_edge_embed:
            self.outer_product_mean = OuterProductMean(c_emb, c_z, c_z//4)
        else:
            self.linear_a_i = Linear(c_emb, c_z)
            self.linear_b_i = Linear(c_emb, c_z)

        self.use_pairwise_dist = use_pairwise_dist
        self.use_radial_basis = use_radial_basis
        self.use_rel_pos = use_rel_pos
        self.use_multimer_rel_pos = use_multimer_rel_pos
        self.one_hot_adj = one_hot_adj
        self.use_topological_distance = use_topological_distance
        self.mask_off_diagonal = mask_off_diagonal

        if self.use_rel_pos:
            self.relpos_k = relpos_k
            self.no_bins = 2 * relpos_k + 1
        if self.use_multimer_rel_pos:
            print('Using multimer relpos.')
            self.max_relative_idx = 32
            self.use_chain_relative = True
            self.max_relative_chain = 2
            if(self.use_chain_relative):
                self.no_bins = (
                    2 * self.max_relative_idx + 2 +
                    1 +
                    2 * self.max_relative_chain + 2
                )
            else:
                self.no_bins = 2 * self.max_relative_idx + 1
        if self.use_rel_pos or self.use_multimer_rel_pos:
            self.linear_relpos = Linear(self.no_bins, c_z)
            if use_topological_distance:
                self.linear_adj = Linear(8, c_z)
            elif self.one_hot_adj:
                self.linear_adj = nn.Embedding(5, c_z)
            else:
                self.linear_adj = Linear(1, c_z)

        if self.use_pairwise_dist:
            if self.use_radial_basis:
                self.linear_dist = nn.Sequential(
                    RadialBasisProjection(64, 1., 21.6875),
                    Linear(64, c_z),
                )
            else:
                self.linear_dist = Linear(1, c_z)

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings
        Implements Algorithm 4.
        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        ) 
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)

    def multimer_relpos(self, ri: torch.Tensor, chain_id: torch.Tensor, entity_id: torch.Tensor, sym_id: torch.Tensor):
        asym_id_same = (chain_id[..., None] == chain_id[..., None, :])
        offset = ri[..., None] - ri[..., None, :]

        clipped_offset = torch.clamp(
            offset + self.max_relative_idx, 0, 2 * self.max_relative_idx
        )

        rel_feats = []
        if(self.use_chain_relative):
            final_offset = torch.where(
                asym_id_same, 
                clipped_offset,
                (2 * self.max_relative_idx + 1) * 
                torch.ones_like(clipped_offset)
            )

            rel_pos = torch.nn.functional.one_hot(
                final_offset,
                2 * self.max_relative_idx + 2,
            )

            rel_feats.append(rel_pos)

            entity_id_same = (entity_id[..., None] == entity_id[..., None, :])
            rel_feats.append(entity_id_same[..., None])

            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]
 
            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain,
                0,
                2 * max_rel_chain,
            )
 
            final_rel_chain = torch.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * max_rel_chain + 1) *
                torch.ones_like(clipped_rel_chain)
            )
 
            rel_chain = torch.nn.functional.one_hot(
                final_rel_chain,
                2 * max_rel_chain + 2,
            )
 
            rel_feats.append(rel_chain)
        else:
            rel_pos = torch.nn.functional.one_hot(
                clipped_offset, 2 * self.max_relative_idx + 1,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(
            self.linear_relpos.weight.dtype
        )

        return self.linear_relpos(rel_feat)

    def forward(
        self,
        aatype: torch.Tensor,
        lig_atom_features: torch.Tensor,
        ti: torch.Tensor,
        edge_mask: torch.Tensor,
        adj: torch.Tensor,
        id_batch: Tuple,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aatype = self.linear_aatype(aatype)
        lig_atom_features = self.linear_lig(lig_atom_features)
        s = torch.cat((aatype, lig_atom_features), -2)

        if self.use_op_edge_embed:
            pair_emb = self.outer_product_mean(s, edge_mask)
        else:
            # [*, N_res, c_z]
            a_i = self.linear_a_i(s)
            b_i = self.linear_b_i(s)
            # [*, N_res, N_res, c_z]
            pair_emb = add( 
                a_i[..., None, :], 
                b_i[..., None, :, :], 
                inplace=inplace_safe
            )
        if self.use_rel_pos or self.use_multimer_rel_pos:
            rel_pos_rec = self.multimer_relpos(*id_batch) if self.use_multimer_rel_pos else self.relpos(id_batch[0].type(s.dtype))
            rel_pos_adj = self.linear_adj(adj)
            rel_pos_rec = torch.cat(
                [rel_pos_rec, torch.zeros(
                    rel_pos_rec.shape[0], rel_pos_rec.shape[1],
                    rel_pos_adj.shape[2], self.c_z, device=rel_pos_rec.device
                )],
                dim = -2
            )
            rel_pos_adj = torch.cat(
                [torch.zeros(
                    rel_pos_adj.shape[0], rel_pos_adj.shape[1],
                    rel_pos_rec.shape[1], self.c_z, device=rel_pos_adj.device
                ), rel_pos_adj],
                dim = -2
            )
            rel_pos = torch.cat([rel_pos_rec, rel_pos_adj], dim=1)
            rel_pos = rel_pos * edge_mask.unsqueeze(-1)
            pair_emb = add(pair_emb, rel_pos, inplace=inplace_safe)
        if self.use_pairwise_dist:
            if self.mask_off_diagonal:
                t_rec = ti[:,:aatype.shape[1]]
                t_lig = ti[:,aatype.shape[1]:]
                rec_dist = torch.cdist(t_rec, t_rec, p=2).unsqueeze(-1).to(dtype=torch.float32)
                lig_dist = torch.cdist(t_lig, t_lig, p=2).unsqueeze(-1).to(dtype=torch.float32)
                rec_dist = self.linear_dist(rec_dist)
                lig_dist = self.linear_dist(lig_dist)
                rec_dist = torch.cat(
                    [rec_dist, torch.zeros(
                        rec_dist.shape[0], rec_dist.shape[1],
                        lig_dist.shape[2], self.c_z, device=rec_dist.device
                    )],
                    dim = -2
                )
                lig_dist = torch.cat(
                    [torch.zeros(
                        lig_dist.shape[0], lig_dist.shape[1],
                        rec_dist.shape[1], self.c_z, device=lig_dist.device
                    ), lig_dist],
                    dim = -2
                )
                pairwise_distance = torch.cat([rec_dist, lig_dist], dim=1)
            else:
                dist = (torch.cdist(ti, ti, p=2) * edge_mask).unsqueeze(-1).to(dtype=torch.float32)
                pairwise_distance = self.linear_dist(dist)
            pair_emb = add(pair_emb, pairwise_distance, inplace=inplace_safe)

        s = self.linear_si(s)
        return s, pair_emb

class RadialBasisProjection(nn.Module):
    def __init__(self, embed_dim: int, min_val: float = 0.0, max_val: float = 2.0):
        super(RadialBasisProjection, self).__init__()
        self.scale = (embed_dim - 1) / (max_val - min_val)
        self.center = nn.Parameter(
            torch.linspace(min_val, max_val, embed_dim), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.scale * torch.square(x - self.center))

class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        # self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )

    @torch.jit.ignore
    def _chunk(self, 
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
        use_memory_efficient_kernel: bool, 
        use_lma: bool,
    ) -> torch.Tensor:
        def fn(m, biases, flash_mask):
            # m = self.layer_norm_m(m)
            return self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
            )

        inputs = {"m": m}
        if(biases is not None):
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        if (self.pair_bias and 
            z is not None and                       # For the 
            self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            chunks = []

            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i: i + 256, :, :]

                # [*, N_res, N_res, C_z]
                z_chunk = self.layer_norm_z(z_chunk)
            
                # [*, N_res, N_res, no_heads]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)
            
            z = torch.cat(chunks, dim=-3)
            
            # [*, 1, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    @torch.jit.ignore
    def _chunked_msa_attn(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_logits: int,
        checkpoint: bool,
        inplace_safe: bool = False
    ) -> torch.Tensor:
        """ 
        MSA attention with training-time chunking of the softmax computation.
        Saves memory in the extra MSA stack. Probably obviated by our fused 
        attention kernel, which is now used by default.
        """
        MSA_DIM = -4

        def _get_qkv(m, z):
            m, mask_bias, z = self._prep_inputs(
                m, z, mask, inplace_safe=inplace_safe
            )
            # m = self.layer_norm_m(m)
            q, k, v = self.mha._prep_qkv(m, m)
            return m, q, k, v, mask_bias, z

        checkpoint_fn = get_checkpoint_fn()

        if(torch.is_grad_enabled() and checkpoint):
            m, q, k, v, mask_bias, z = checkpoint_fn(_get_qkv, m, z)
        else:
            m, q, k, v, mask_bias, z = _get_qkv(m, z)
       
        o = _attention_chunked_trainable(
            query=q, 
            key=k, 
            value=v, 
            biases=[mask_bias, z], 
            chunk_size=chunk_logits, 
            chunk_dim=MSA_DIM,
            checkpoint=checkpoint,
        )

        if(torch.is_grad_enabled() and checkpoint):
            # Storing an additional m here is far from ideal
            m = checkpoint_fn(self.mha._wrap_up, o, m)
        else:
            m = self.mha._wrap_up(o, m)

        return m

    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _chunk_logits: Optional[int] = None,
        _checkpoint_chunks: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
                
        """
        if(_chunk_logits is not None):
            return self._chunked_msa_attn(
                m=m, z=z, mask=mask, 
                chunk_logits=_chunk_logits, 
                checkpoint=_checkpoint_chunks,
                inplace_safe=inplace_safe,
            )
       
        m, mask_bias, z = self._prep_inputs(
            m, z, mask, inplace_safe=inplace_safe
        )

        biases = [mask_bias]
        if(z is not None):
            biases.append(z)

        if chunk_size is not None:
            m = self._chunk(
                m, 
                biases, 
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel, 
                use_lma=use_lma,
            )
        else:
            m = self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
            )

        return m

class MSAAttention_w_LayerNorm(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention_w_LayerNorm, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)
        # maybe this will work:
        # self.layer_norm_m = nn.LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )

    @torch.jit.ignore
    def _chunk(self, 
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
        use_memory_efficient_kernel: bool, 
        use_lma: bool,
    ) -> torch.Tensor:
        def fn(m, biases, flash_mask):
            m = self.layer_norm_m(m)
            return self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
            )

        inputs = {"m": m}
        if(biases is not None):
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        if (self.pair_bias and 
            z is not None and                       # For the 
            self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            chunks = []

            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i: i + 256, :, :]

                # [*, N_res, N_res, C_z]
                z_chunk = self.layer_norm_z(z_chunk)
            
                # [*, N_res, N_res, no_heads]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)
            
            z = torch.cat(chunks, dim=-3)
            
            # [*, 1, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    @torch.jit.ignore
    def _chunked_msa_attn(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_logits: int,
        checkpoint: bool,
        inplace_safe: bool = False
    ) -> torch.Tensor:
        """ 
        MSA attention with training-time chunking of the softmax computation.
        Saves memory in the extra MSA stack. Probably obviated by our fused 
        attention kernel, which is now used by default.
        """
        MSA_DIM = -4

        def _get_qkv(m, z):
            m, mask_bias, z = self._prep_inputs(
                m, z, mask, inplace_safe=inplace_safe
            )
            m = self.layer_norm_m(m)
            q, k, v = self.mha._prep_qkv(m, m)
            return m, q, k, v, mask_bias, z

        checkpoint_fn = get_checkpoint_fn()

        if(torch.is_grad_enabled() and checkpoint):
            m, q, k, v, mask_bias, z = checkpoint_fn(_get_qkv, m, z)
        else:
            m, q, k, v, mask_bias, z = _get_qkv(m, z)
       
        o = _attention_chunked_trainable(
            query=q, 
            key=k, 
            value=v, 
            biases=[mask_bias, z], 
            chunk_size=chunk_logits, 
            chunk_dim=MSA_DIM,
            checkpoint=checkpoint,
        )

        if(torch.is_grad_enabled() and checkpoint):
            # Storing an additional m here is far from ideal
            m = checkpoint_fn(self.mha._wrap_up, o, m)
        else:
            m = self.mha._wrap_up(o, m)

        return m

    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _chunk_logits: Optional[int] = None,
        _checkpoint_chunks: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
                
        """
        if(_chunk_logits is not None):
            return self._chunked_msa_attn(
                m=m, z=z, mask=mask, 
                chunk_logits=_chunk_logits, 
                checkpoint=_checkpoint_chunks,
                inplace_safe=inplace_safe,
            )
       
        m, mask_bias, z = self._prep_inputs(
            m, z, mask, inplace_safe=inplace_safe
        )

        biases = [mask_bias]
        if(z is not None):
            biases.append(z)

        if chunk_size is not None:
            m = self._chunk(
                m, 
                biases, 
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel, 
                use_lma=use_lma,
            )
        else:
            m = self.layer_norm_m(m)
            m = self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
            )

        return m

class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )

class MSARowAttentionWithPairBias_w_LayerNorm(MSAAttention_w_LayerNorm):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias_w_LayerNorm, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )
        
class EvoformerBlockCore(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
        _is_extra_msa_stack: bool = False,
        opm_first: bool=False,
    ):
        super(EvoformerBlockCore, self).__init__()

        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )

        if not opm_first:
            self.outer_product_mean = OuterProductMean(
                c_m,
                c_z,
                c_hidden_opm,
            )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        # self.tri_att_start = TriangleAttention(
        #     c_z,
        #     c_hidden_pair_att,
        #     no_heads_pair,
        #     inf=inf,
        # )
        # self.tri_att_end = TriangleAttention(
        #     c_z,
        #     c_hidden_pair_att,
        #     no_heads_pair,
        #     inf=inf,
        # )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

        self.opm_first = opm_first

    def forward(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        msa_trans_mask = msa_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None

        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        m, z = input_tensors

        m = add(
            m,
            self.msa_transition(
                m, mask=msa_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        ) 

        if(_offload_inference and inplace_safe):
            del m, z
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            m, z = input_tensors 

        if not self.opm_first:
            opm = self.outer_product_mean(
                m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
            )

        if(_offload_inference and inplace_safe):
            del m, z
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(opm.device)
            m, z = input_tensors

        if not self.opm_first:
            z = add(z, opm, inplace=inplace_safe)
            del opm

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        z = tmu_update if inplace_safe else z + self.ps_dropout_row_layer(tmu_update)
        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        z = tmu_update if inplace_safe else z + self.ps_dropout_row_layer(tmu_update)
        del tmu_update

        # z = add(z, 
        #     self.ps_dropout_row_layer(
        #         self.tri_att_start(
        #             z, 
        #             mask=pair_mask, 
        #             chunk_size=_attn_chunk_size, 
        #             use_lma=use_lma,
        #             inplace_safe=inplace_safe,
        #         ),
        #     ),
        #     inplace=inplace_safe,
        # )

        # z = z.transpose(-2, -3)
        # if(inplace_safe):
        #     input_tensors[1] = z.contiguous()
        #     z = input_tensors[1]

        # z = add(z,
        #     self.ps_dropout_row_layer(
        #         self.tri_att_end(
        #             z,
        #             mask=pair_mask.transpose(-1, -2),
        #             chunk_size=_attn_chunk_size,
        #             use_lma=use_lma,
        #             inplace_safe=inplace_safe,
        #         ),
        #     ),
        #     inplace=inplace_safe,
        # )

        # z = z.transpose(-2, -3)

        if(inplace_safe):
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]

        z = z + self.pair_transition(
                z, mask=pair_trans_mask, chunk_size=chunk_size,
            )

        if(_offload_inference and inplace_safe):
            device = z.device
            del m, z
            input_tensors[0] = input_tensors[0].to(device)
            input_tensors[1] = input_tensors[1].to(device)
            m, z = input_tensors

        return m, z

class EvoformerBlockCore_w_TriangleAttention(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
        _is_extra_msa_stack: bool = False,
        opm_first: bool=False,
    ):
        super(EvoformerBlockCore_w_TriangleAttention, self).__init__()

        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )

        if not opm_first:
            self.outer_product_mean = OuterProductMean(
                c_m,
                c_z,
                c_hidden_opm,
            )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

        self.opm_first = opm_first

    def forward(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        msa_trans_mask = msa_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None

        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        m, z = input_tensors

        m = add(
            m,
            self.msa_transition(
                m, mask=msa_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        ) 

        if(_offload_inference and inplace_safe):
            del m, z
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            m, z = input_tensors 

        if not self.opm_first:
            opm = self.outer_product_mean(
                m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
            )

        if(_offload_inference and inplace_safe):
            del m, z
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(opm.device)
            m, z = input_tensors

        if not self.opm_first:
            z = add(z, opm, inplace=inplace_safe)
            del opm

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        z = tmu_update if inplace_safe else z + self.ps_dropout_row_layer(tmu_update)
        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        z = tmu_update if inplace_safe else z + self.ps_dropout_row_layer(tmu_update)
        del tmu_update

        z = add(z, 
            self.ps_dropout_row_layer(
                self.tri_att_start(
                    z, 
                    mask=pair_mask, 
                    chunk_size=_attn_chunk_size, 
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                ),
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if(inplace_safe):
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]

        z = add(z,
            self.ps_dropout_row_layer(
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2),
                    chunk_size=_attn_chunk_size,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                ),
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)

        if(inplace_safe):
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]

        z = z + self.pair_transition(
                z, mask=pair_trans_mask, chunk_size=chunk_size,
            )

        if(_offload_inference and inplace_safe):
            device = z.device
            del m, z
            input_tensors[0] = input_tensors[0].to(device)
            input_tensors[1] = input_tensors[1].to(device)
            m, z = input_tensors

        return m, z

class EvoformerBlock(nn.Module):
    def __init__(self,
        c_s: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        opm_first: bool=False,
    ):
        super(EvoformerBlock, self).__init__()

        if opm_first:
            self.outer_product_mean = OuterProductMean(
                c_s,
                c_z,
                c_hidden_opm,
            )

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_s,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.core = EvoformerBlockCore(
            c_m=c_s,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps,
            opm_first=opm_first,
        )

        self.opm_first = opm_first

    def forward(self,
        s: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        if self.opm_first:
            opm = self.outer_product_mean(
            s, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
            )
            z = add(z, opm, inplace=inplace_safe)

        s = add(s, 
            self.msa_dropout_layer(
                self.msa_att_row(
                    s, 
                    z=z, 
                    mask=msa_mask, 
                    chunk_size=_attn_chunk_size,
                    use_lma=use_lma,
                    use_memory_efficient_kernel=False,
                    _chunk_logits=_attn_chunk_size,
                )
            ),
            inplace=inplace_safe,
        )

        input_tensors = [s, z]
        
        del s, z

        s, z = self.core(
            input_tensors, 
            msa_mask=msa_mask, 
            pair_mask=pair_mask, 
            chunk_size=chunk_size, 
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
            _offload_inference=_offload_inference,
        )

        return s, z

class FullEvoformerBlock(nn.Module):
    def __init__(self,
        c_s: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        opm_first: bool=False,
    ):
        super(FullEvoformerBlock, self).__init__()

        if opm_first:
            self.outer_product_mean = OuterProductMean(
                c_s,
                c_z,
                c_hidden_opm,
            )

        self.msa_att_row = MSARowAttentionWithPairBias_w_LayerNorm(
            c_m=c_s,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.core = EvoformerBlockCore_w_TriangleAttention(
            c_m=c_s,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps,
            opm_first=opm_first,
        )

        self.opm_first = opm_first

    def forward(self,
        s: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        if self.opm_first:
            opm = self.outer_product_mean(
            s, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
            )
            z = add(z, opm, inplace=inplace_safe)

        s = add(s, 
            self.msa_dropout_layer(
                self.msa_att_row(
                    s, 
                    z=z, 
                    mask=msa_mask, 
                    chunk_size=_attn_chunk_size,
                    use_lma=use_lma,
                    use_memory_efficient_kernel=False,
                    _chunk_logits=_attn_chunk_size,
                )
            ),
            inplace=inplace_safe,
        )

        input_tensors = [s, z]
        
        del s, z

        s, z = self.core(
            input_tensors, 
            msa_mask=msa_mask, 
            pair_mask=pair_mask, 
            chunk_size=chunk_size, 
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
            _offload_inference=_offload_inference,
        )

        return s, z
    
class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s_out: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_evo_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        blocks_per_ckpt: int=1,
        inf: float = 1e5,
        eps: float = 1e-8,
        opm_first: bool=False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s_out:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_evo_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
        """
        super(EvoformerStack, self).__init__()
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList([
            EvoformerBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
                opm_first=opm_first,
            ) for _ in range(no_evo_blocks)
        ])

        self.linear = Linear(c_s, c_s_out)

    def forward(self,
        s: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[torch.Tensor] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_lma: Whether to use low-memory attention during inference
        Returns:
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """ 
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        def block_with_cache_clear(block, *args, **kwargs):
            torch.cuda.empty_cache()
            return block(*args, **kwargs)

        blocks = [partial(block_with_cache_clear, b) for b in blocks]

        blocks_per_ckpt = self.blocks_per_ckpt
        if(not torch.is_grad_enabled()):
            blocks_per_ckpt = None
        
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        s = self.linear(s[..., 0, :, :])

        return s, z

class FullEvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s_out: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_evo_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        blocks_per_ckpt: int=1,
        inf: float = 1e5,
        eps: float = 1e-8,
        opm_first: bool=False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s_out:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_evo_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
        """
        super(FullEvoformerStack, self).__init__()
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList([
            FullEvoformerBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
                opm_first=opm_first,
            ) for _ in range(no_evo_blocks)
        ])

        self.linear = Linear(c_s, c_s_out)

    def forward(self,
        s: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[torch.Tensor] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_lma: Whether to use low-memory attention during inference
        Returns:
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """ 
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        def block_with_cache_clear(block, *args, **kwargs):
            torch.cuda.empty_cache()
            return block(*args, **kwargs)

        blocks = [partial(block_with_cache_clear, b) for b in blocks]

        blocks_per_ckpt = self.blocks_per_ckpt
        if(not torch.is_grad_enabled()):
            blocks_per_ckpt = None
        
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        s = self.linear(s[..., 0, :, :])

        return s, z

class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, sum_pool = False, mean_pool = False, att_update=False, construct_frames=False):
        """
        Args:
            c_s:
                Single representation channel dimension
            sum_pool:
                Sum pooling.
            mean_pool:
                Mean pooling.
            att_update:
                Use an intermolecular attention layer.
            construct_frames:
                Whether or not QuickBind is run with frames. Only used to set the output dimension.
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self.sum_pool = sum_pool
        self.mean_pool = mean_pool
        self.att_update = att_update

        if self.att_update:
            self.mha = Attention(
                self.c_s, 
                self.c_s, 
                self.c_s, 
                c_hidden = self.c_s, 
                no_heads = 4,
            )
        
        final_dim = 6 if construct_frames else 3

        if self.sum_pool or self.mean_pool:
            self.linear = Linear(2*self.c_s, final_dim, init="final")
        else:
            self.linear = Linear(self.c_s, final_dim, init="final")

    def forward(self, s: torch.Tensor, rec_mask: torch.Tensor, lig_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 3] update vector 
        """
        if self.sum_pool:
            s_protein = s[:, :rec_mask.shape[-1], :]
            s_ligand = s[:, rec_mask.shape[-1]:, :]
            s_protein = torch.sum(s_protein, dim=-2, keepdim=True)
            s = torch.cat([s_ligand, s_protein.repeat(1, s_ligand.shape[1], 1)], dim=-1)
        if self.mean_pool:
            s_protein = s[:, :rec_mask.shape[-1], :]
            s_protein = s_protein * rec_mask.unsqueeze(-1)
            s_ligand = s[:, rec_mask.shape[-1]:, :]
            s_protein = torch.mean(s_protein, dim=-2, keepdim=True) / (torch.sum(rec_mask, dim=-1, keepdim=True)).unsqueeze(-1)
            s = torch.cat([s_ligand, s_protein.repeat(1, s_ligand.shape[1], 1)], dim=-1)
        if self.att_update:
            s_protein = s[:, :rec_mask.shape[-1], :]
            s_ligand = s[:, rec_mask.shape[-1]:, :]
            protein_bias = self.mha(q_x=s_ligand, kv_x=s_protein)
            s = s_ligand + protein_bias
        # YET ANOTHER MODALITY
        # s_protein = s_protein.unsqueeze(-2).repeat(1, 1, s_ligand.shape[1], 1)
        # s_ligand = s_ligand.unsqueeze(1).repeat(1, s_protein.shape[1], 1, 1)
        # s = s_protein * s_ligand
        # equivalent to:
        # s = torch.einsum('bic,bjc->bijc', s_protein, s_ligand)
        # s = torch.sum(s, dim = 1)
        # [*, 3]
        # update = update * lig_mask.unsqueeze(-1)

        return self.linear(s) 

class StructureModule(nn.Module):
    def __init__(
        self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, dropout_rate,
        no_transition_layers, sum_pool, mean_pool, att_update, use_gated_ipa = False, construct_frames = False
    ):
        super(StructureModule, self).__init__()
        if use_gated_ipa:
            self.ipa = GatedInvariantPointAttention(
                c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points
            )
        else:
            self.ipa = InvariantPointAttention(
                c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points
            )
        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa = LayerNorm(c_s)
        self.transition = StructureModuleTransition(c_s, no_transition_layers, dropout_rate)
        self.bb_update = BackboneUpdate(c_s, sum_pool, mean_pool, att_update, construct_frames)
    
    def forward(self, s, z, rigids, mask, rec_mask, lig_mask):
        s = s + self.ipa(s, z, rigids, mask)
        s = self.ipa_dropout(s)
        s = self.layer_norm_ipa(s)
        s = self.transition(s)
        new_trans = self.bb_update(s, rec_mask, lig_mask)
        return s, z, new_trans

class GatedInvariantPointAttention(nn.Module):
    """
    Implements a gated variant of Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(GatedInvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
                self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.attn = nn.Softmax(dim=-1)

        self.softplus = nn.Softplus()

        self.linear_g = Linear(
            self.c_s, concat_out_dim, init="gating"
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] affine transformation object
            edge_mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts) ########################################################

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts) ########################################################

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))

        # [*, N_res, N_res, H]
        b = self.linear_b(z)
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att ** 2
        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights
        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att

        # [*, N_res, N_res]
        square_mask = self.inf * (edge_mask - 1)
        a = a + square_mask.unsqueeze(-3)
        a = self.attn(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # As DeepMind explains, this manual matmul ensures that the operation
        # happens in float32.
        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt) ########################################################

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        final_o = torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z.dtype)

        g = self.sigmoid(self.linear_g(s))
        final_o = final_o * g

        # [*, N_res, C_s]
        s = self.linear_out(final_o)

        return s
