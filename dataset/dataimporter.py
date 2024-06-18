import os
import pathlib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .process_mols import get_receptor_input, read_molecule, process_ligand
from commons.utils import read_strings_from_txt, log
from torch_geometric.data import Data
from openfold.utils.rigid_utils import Rotation, Rigid
torch.cuda.empty_cache()

class DataImporter(Dataset):

    def __init__(
        self,
        complex_names_path='~/ProteinLigandBinding/data/timesplit_no_lig_overlap_train',
        chain_radius=10,
        remove_h=True,
        cropping=True,
        crop_size=512,
        recenter=True,
        binding_site_cropping=True,
        spatial_cropping=False,
        spatial_and_contig_cropping=False,
        blackhole_init=False,
        count_repeats=False,
        rand_lig_coords=False,
        unseen_only=False,
        seed=0
    ):
        """"
        complex_names_path:
            Path to file with names of the directories containing the input files.
        chain_radius:
            Maximum distance to a ligand atom for a protein chain to still be included. 
        remove_h:
            Whether or not to explicitely consider ligand H atoms.
        cropping:
            Whether or not to crop the protein.
        crop_size:
            Number of amino acids to crop the protein to.
        recenter:
            Whether or not the input protein should be recentred first.
        binding_site_cropping:
            Whether or not to do binding site cropping.
        spatial_cropping:
            Whether or not to do spatial cropping.
        spatial_and_contig_cropping:
            Whether or not to do randomly do spatial or contiguos cropping.
        blackhole_init:
            Whether or not to do black hole initialisation of ligand atom coordinates (rather than using an RDKit conformer).
        count_repeats:
            If using black hole initialisation, whether or not to index atoms that have the same chemical properties.
        rand_lig_coords:
            Whether or not do randomly rotate the ligand first.
        unseen_only:
            Whether or not to only load unseen PDBBind proteins.
        """
        
        self.complex_names_path = complex_names_path
        self.chain_radius = chain_radius
        self.remove_h = remove_h
        self.cropping = cropping
        self.crop_size = crop_size
        self.recenter = recenter
        self.binding_site_cropping = binding_site_cropping
        self.spatial_cropping = spatial_cropping
        self.spatial_and_contig_cropping = spatial_and_contig_cropping
        self.blackhole_init = blackhole_init
        self.count_repeats = count_repeats
        self.rand_lig_coords = rand_lig_coords
        self.unseen_only = unseen_only
        self.seed = seed

        # get and set some useful paths
        self.directory = pathlib.Path(__file__).parent.resolve()
        if not os.path.basename(self.complex_names_path) == 'posebusters':
            self.dataset_dir = os.path.join(self.directory, '../data/PDBBind')
        else:
            self.dataset_dir = os.path.join(self.directory, '../data/posebusters_benchmark_set')
        self.processed_dir = os.path.join(self.directory, '../data/processed/', os.path.basename(self.complex_names_path))
        if not os.path.exists(os.path.join(self.directory, '../data/processed/')):
            os.mkdir(os.path.join(self.directory, '../data/processed/'))
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

        # process data if not already done
        if (
            not os.path.exists(os.path.join(self.processed_dir, 'rec_input_chain_ids.pt')) or
            (not os.path.exists(os.path.join(self.processed_dir, 'lig_input_framing.pt')) and not self.remove_h) or
            (not os.path.exists(os.path.join(self.processed_dir, 'lig_input_framing_noH.pt')) and self.remove_h)
        ):
            self._process()
        if not os.path.exists(os.path.join(self.processed_dir, 'rec_input_proc_ids.pt')):
            self._process_chain_ids()
        if (
            self.unseen_only and
            (
            (not os.path.exists(os.path.join(self.processed_dir, 'unseen_lig_input_framing.pt')) and not self.remove_h) or
            (not os.path.exists(os.path.join(self.processed_dir, 'unseen_lig_input_framing_noH.pt')) and self.remove_h) or
            not os.path.exists(os.path.join(self.processed_dir, 'unseen_rec_input_proc_ids.pt'))
            )
        ):
            self._process_unseen()

        # load data into memory
        log('Loading data into memory.')
        if self.remove_h:
            self.ligands = torch.load(os.path.join(self.processed_dir, f"{'unseen_' if self.unseen_only else ''}lig_input_framing_noH.pt"))
        else:
            self.ligands = torch.load(os.path.join(self.processed_dir, f"{'unseen_' if self.unseen_only else ''}lig_input_framing.pt"))
        self.receptors = torch.load(os.path.join(self.processed_dir, f"{'unseen_' if self.unseen_only else ''}rec_input_proc_ids.pt"))

        # recentring and cropping
        if self.recenter:
            self.receptors, self.ligands = self._recenter_proteins_and_ligands(self.receptors, self.ligands)
        if self.cropping:
            log(f'Cropping sequences into chunks of {self.crop_size} residues.')
        elif to_rem := [
            idx
            for idx, rec in enumerate(self.receptors)
            if len(rec['c_alpha_coords']) > 2000
        ]:
            for idx in sorted(to_rem, reverse=True):
                removed = self.receptors.pop(idx)
                del self.ligands[idx]
                removed_name = removed['complex_names']
                log(f'Removed complex {removed_name} because it contains more than 2000 residues.')
            log(f'Removed {len(to_rem)} complexes in total.')
        if self.binding_site_cropping and self.cropping:
            log('Finding binding site residues for binding site cropping.')
            indices = self._find_binding_site_residues(self.receptors, self.ligands)
            log('Cropping proteins.')
            self.receptors = self._crop_all_to_size(self.receptors, indices, self.crop_size)
        if self.spatial_cropping and self.cropping:
            log('Using spatial cropping, finding binding site residues.')
            indices = self._find_all_binding_site_residues(self.receptors, self.ligands, self.crop_size)
            log('Cropping proteins.')
            self.receptors = self._non_contig_crop_to_size(self.receptors, indices, self.crop_size)
        if self.spatial_and_contig_cropping and self.cropping:
            log('Using spatial and contiguous cropping, finding binding site residues.')
            self.indices = self._find_all_binding_site_residues(self.receptors, self.ligands, self.crop_size)

        assert len(self.ligands) == len(self.receptors)
        log('Finished loading data into memory.')
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.ligands)
    
    def __getitem__(self, idx):
        ligand = self.ligands[idx]
        receptor = self.receptors[idx]
        seq_length = receptor['seq_length'].item()

        # cropping
        if (
            self.cropping and not (
                self.binding_site_cropping or self.spatial_cropping or self.spatial_and_contig_cropping
            )
        ):
            aatype, c_alpha_coords, n_coords, c_coords, ri, \
                chain_ids_processed, entity_ids_processed, sym_ids_processed = self._random_crop_to_size(
                receptor['aatype'], receptor['c_alpha_coords'],
                receptor['n_coords'], receptor['c_coords'],
                receptor['residue_index'], self.crop_size,
                seq_length, receptor['chain_ids_processed'],
                receptor['entity_ids_processed'], receptor['sym_ids_processed'],
            )

        elif self.cropping and self.spatial_and_contig_cropping:
            use_spatial_cropping = bool(torch.randint(0, 2, (1,)))
            if use_spatial_cropping:
                aatype, c_alpha_coords, n_coords, c_coords, ri, \
                    chain_ids_processed, entity_ids_processed, sym_ids_processed = self._crop_to_size(
                    receptor['aatype'], receptor['c_alpha_coords'],
                    receptor['n_coords'], receptor['c_coords'],
                    receptor['residue_index'], self.crop_size,
                    seq_length, receptor['chain_ids_processed'],
                    receptor['entity_ids_processed'], receptor['sym_ids_processed'],
                    self.indices[idx],
                )
            
            else:
                aatype, c_alpha_coords, n_coords, c_coords, ri, \
                    chain_ids_processed, entity_ids_processed, sym_ids_processed = self._random_crop_to_size(
                    receptor['aatype'], receptor['c_alpha_coords'],
                    receptor['n_coords'], receptor['c_coords'],
                    receptor['residue_index'], self.crop_size,
                    seq_length, receptor['chain_ids_processed'],
                    receptor['entity_ids_processed'], receptor['sym_ids_processed'],
                )

        else:
            aatype, c_alpha_coords, n_coords, c_coords, ri, \
                    chain_ids_processed, entity_ids_processed, sym_ids_processed =\
                    receptor['aatype'], receptor['c_alpha_coords'], receptor['n_coords'], \
                    receptor['c_coords'], receptor['residue_index'], receptor['chain_ids_processed'], \
                    receptor['entity_ids_processed'], receptor['sym_ids_processed']

        # random ligand transformation
        c_alpha_coords, n_coords, c_coords, \
            lig_atom_coords, pseudo_N, pseudo_C, \
                true_lig_atom_coords, true_pseudo_N, true_pseudo_C = self._random_transform(
            c_alpha_coords, n_coords, c_coords,
            ligand['atom_coords'], ligand['pseudo_N'], ligand['pseudo_C'],
            ligand['true_atom_coords'], ligand['true_pseudo_N'], ligand['true_pseudo_C']
        ) if self.rand_lig_coords else (
            c_alpha_coords, n_coords, c_coords,
            ligand['atom_coords'], ligand['pseudo_N'], ligand['pseudo_C'],
            ligand['true_atom_coords'], ligand['true_pseudo_N'], ligand['true_pseudo_C']
        )

        # black hole initialisation
        lig_atom_features = ligand['atom_features']
        if self.blackhole_init:
            lig_atom_coords = torch.zeros_like(lig_atom_coords)
            if self.count_repeats:
                repeat_idx = self._count_atom_repeats(ligand['atom_features'])
                lig_atom_features = torch.cat((lig_atom_features, repeat_idx), -1)

        # return data
        data = Data(
            complex_name = receptor['complex_names'],
            aatype = aatype, residue_index = ri,
            chain_ids_processed = chain_ids_processed,
            entity_ids_processed = entity_ids_processed,
            sym_ids_processed = sym_ids_processed,
            c_alpha_coords = c_alpha_coords,
            n_coords = n_coords,
            c_coords = c_coords,
            lig_atom_features = lig_atom_features,
            pseudo_N = pseudo_N,
            pseudo_C = pseudo_C,
            true_pseudo_N = true_pseudo_N,
            true_pseudo_C = true_pseudo_C,
            adjacency = ligand['adjacency'],
            lig_atom_coords = lig_atom_coords,
            true_lig_atom_coords = true_lig_atom_coords,
            distance_matrix = ligand['distance_matrix'],
            adjacency_bo = ligand['adjacency_bo'],
        )
        del ligand, receptor
        return data
    
    # helper functions
    def _random_crop_to_size(self, aatype, c_alpha_coords, n_coords, c_coords, ri, crop_size, seq_length, chain_ids, entity_ids, sym_ids):
        """Crop randomly to `crop_size`, or keep as is if shorter than that."""
        g = torch.Generator(device=aatype.device)
        num_res_crop_size = min(int(seq_length), crop_size)
    
        def _randint(lower, upper):
            return int(torch.randint(
                    lower,
                    upper + 1,
                    (1,),
                    device=aatype.device,
                    generator=g,
            )[0])
    
        n = seq_length - num_res_crop_size
        crop_start = _randint(0, n)
        aatype_sliced = aatype[crop_start:(crop_start+crop_size)]
        c_alpha_coords_sliced = c_alpha_coords[crop_start:(crop_start+crop_size)]
        n_coords_sliced = n_coords[crop_start:(crop_start+crop_size)]
        c_coords_sliced = c_coords[crop_start:(crop_start+crop_size)]
        ri_sliced = ri[crop_start:(crop_start+crop_size)]
        chain_ids_sliced = chain_ids[crop_start:(crop_start+crop_size)]
        entity_ids_sliced = entity_ids[crop_start:(crop_start+crop_size)]
        sym_ids_sliced = sym_ids[crop_start:(crop_start+crop_size)]
        
        return aatype_sliced, c_alpha_coords_sliced, n_coords_sliced, c_coords_sliced, ri_sliced, chain_ids_sliced, entity_ids_sliced, sym_ids_sliced

    def _find_binding_site_residues(self, receptors, ligands):
        indices = []
        for receptor, ligand in tqdm(zip(receptors, ligands), total=len(receptors)):
            c_alpha_coords = receptor['c_alpha_coords']
            lig_coords = ligand['true_atom_coords']
            distances = torch.cdist(c_alpha_coords.to(dtype=torch.float32), lig_coords.to(dtype=torch.float32))
            indices.append(torch.argmin(torch.min(distances, dim=1).values))
        return indices
    
    def _find_all_binding_site_residues(self, receptors, ligands, crop_size):
        indices = []
        for receptor, ligand in tqdm(zip(receptors, ligands), total=len(receptors)):
            c_alpha_coords = receptor['c_alpha_coords']
            lig_coords = ligand['true_atom_coords']
            distances = torch.cdist(c_alpha_coords.to(dtype=torch.float32), lig_coords.to(dtype=torch.float32))
            distances_flattened = torch.min(distances, dim=1).values
            _, idx = distances_flattened.sort()
            indices.append(idx[:crop_size])
        return indices

    def _crop_all_to_size(self, receptors, indices, crop_size):
        assert len(receptors) == len(indices)
        for receptor, idx in tqdm(zip(receptors, indices), total=len(receptors)):
            seq_length = receptor['seq_length'].item()
            if seq_length < crop_size:
                aatype_sliced = receptor['aatype']
                c_alpha_coords_sliced = receptor['c_alpha_coords']
                n_coords_sliced = receptor['n_coords']
                c_coords_sliced = receptor['c_coords']
                ri_sliced = receptor['residue_index']
            else:
                # get start and end indices, making sure that we crop to the full crop_size, if possible
                start = idx - crop_size/2
                end = idx + crop_size/2
                if start < 0:
                    end -= start
                    start = 0
                if end > seq_length:
                    start -=  (end - seq_length)
                    end = seq_length
                start = int(start)
                end = int(end)
                aatype_sliced = receptor['aatype'][start:end]
                c_alpha_coords_sliced = receptor['c_alpha_coords'][start:end]
                n_coords_sliced = receptor['n_coords'][start:end]
                c_coords_sliced = receptor['c_coords'][start:end]
                ri_sliced = receptor['residue_index'][start:end]
                        
            receptor['aatype'] = aatype_sliced
            receptor['c_alpha_coords'] = c_alpha_coords_sliced
            receptor['n_coords'] = n_coords_sliced
            receptor['c_coords'] = c_coords_sliced
            receptor['residue_index'] = ri_sliced

        return receptors

    def _crop_to_size(self, aatype, c_alpha_coords, n_coords, c_coords, ri, crop_size, seq_length, chain_ids, entity_ids, sym_ids, indices):
        if seq_length < crop_size:
            return (
                aatype,
                c_alpha_coords,
                n_coords,
                c_coords,
                ri,
                chain_ids,
                entity_ids,
                sym_ids,
            )
        else:
            return(
                aatype[indices],
                c_alpha_coords[indices],
                n_coords[indices],
                c_coords[indices],
                ri[indices],
                chain_ids[indices],
                entity_ids[indices],
                sym_ids[indices],
            )

    def _non_contig_crop_to_size(self, receptors, indices, crop_size):
        for receptor, idx in tqdm(zip(receptors, indices), total=len(receptors)):
            seq_length = receptor['seq_length'].item()
            if seq_length < crop_size:
                aatype_sliced = receptor['aatype']
                c_alpha_coords_sliced = receptor['c_alpha_coords']
                n_coords_sliced = receptor['n_coords']
                c_coords_sliced = receptor['c_coords']
                ri_sliced = receptor['residue_index']
            else:                    
                aatype_sliced = receptor['aatype'][idx]
                c_alpha_coords_sliced = receptor['c_alpha_coords'][idx]
                n_coords_sliced = receptor['n_coords'][idx]
                c_coords_sliced = receptor['c_coords'][idx]
                ri_sliced = receptor['residue_index'][idx]
                    
            receptor['aatype'] = aatype_sliced
            receptor['c_alpha_coords'] = c_alpha_coords_sliced
            receptor['n_coords'] = n_coords_sliced
            receptor['c_coords'] = c_coords_sliced
            receptor['residue_index'] = ri_sliced

        return receptors
    
    def _recenter_proteins_and_ligands(self, receptors, ligands):
        for rec, lig in zip(receptors, ligands):
            centre_of_mass = torch.mean(rec['c_alpha_coords'], dim=0)
            new_c_alpha_coords = rec['c_alpha_coords'] - centre_of_mass
            new_n_coords = rec['n_coords'] - centre_of_mass
            new_c_coords = rec['c_coords'] - centre_of_mass
            new_true_atom_coords = lig['true_atom_coords'] - centre_of_mass
            new_true_pseudo_N = lig['true_pseudo_N'] - centre_of_mass
            new_true_pseudo_C = lig['true_pseudo_C'] - centre_of_mass
            rec['c_alpha_coords'] = new_c_alpha_coords
            rec['n_coords'] = new_n_coords
            rec['c_coords'] = new_c_coords
            lig['true_atom_coords'] = new_true_atom_coords
            lig['true_pseudo_N'] = new_true_pseudo_N
            lig['true_pseudo_C'] = new_true_pseudo_C

        return receptors, ligands

    def _count_atom_repeats(self, atom_features):
        repeat_indx = torch.zeros(atom_features.shape[0]).unsqueeze(-1)
        count_dict = {}
        for i, a_i in enumerate(atom_features):
            indx = 1 
            a_i = a_i[0].item() # atomic number
            if a_i in count_dict: 
                indx = count_dict[a_i] + 1
            count_dict[a_i] = indx 
            repeat_indx[i] = indx 
        return repeat_indx

    def _get_rand_rotation_matrix(self):
        import numpy as np

        randnums = np.random.uniform(size=(3,))
        theta, phi, z = randnums
        theta = theta * 2.0* np.pi
        phi = phi * 2.0 * np.pi 
        z = z * 2.0
        
        r = np.sqrt(z)
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
            )
        st = np.sin(theta)
        ct = np.cos(theta)
        
        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
        M = (np.outer(V, V) - np.eye(3)).dot(R)

        return torch.tensor(M)

    def _random_transform(
        self, c_alpha_coords, n_coords, c_coords,
        lig_coords, pseudo_N, pseudo_C,
        true_lig_coords, true_pseudo_N, true_pseudo_C
    ):
        # rot_mats = self._get_rand_rotation_matrix()
        # T = Rotation(rot_mats=rot_mats, quats=None)
        g = torch.Generator(device=c_alpha_coords.device)
        T = Rigid(
            Rotation(rot_mats=torch.rand(3, 3, generator=g), quats=None), 25+torch.rand(3,  generator=g)
        )
        # c_alpha_coords = T.apply(c_alpha_coords)
        # n_coords = T.apply(n_coords)
        # c_coords = T.apply(c_coords)
        lig_coords = T.apply(lig_coords)
        pseudo_N = T.apply(pseudo_N)
        pseudo_C = T.apply(pseudo_C)
        # true_lig_coords = T.apply(true_lig_coords)
        # true_pseudo_N = T.apply(true_pseudo_N)
        # true_pseudo_C = T.apply(true_pseudo_C)
        return (
            c_alpha_coords, n_coords, c_coords,
            lig_coords, pseudo_N, pseudo_C,
            true_lig_coords, true_pseudo_N, true_pseudo_C
        )

    def get_feature_dimensions(self):
        lig, rec = self.ligands[0], self.receptors[0]
        lig_feat_dim = lig['atom_features'].size(-1) + 1 if self.count_repeats else lig['atom_features'].size(-1)
        rec_feat_dim = rec['aatype'].size(-1)
        return lig_feat_dim, rec_feat_dim

    def _process(self):
        log(f'Processing complexes from [{self.complex_names_path}] and saving them to [{self.processed_dir}].')
        complex_names = read_strings_from_txt(self.complex_names_path)
        if '1xn3' in complex_names:
            complex_names.remove('1xn3') # corrupt PDB file
        log(f'Loading {len(complex_names)} complexes.')

        ligs = []
        if os.path.basename(self.complex_names_path) == 'posebusters':
            for name in tqdm(complex_names, desc='Loading ligands'):
                lig = read_molecule(os.path.join(self.dataset_dir, name, f'{name}_ligand.sdf'), remove_hs=self.remove_h)
                ligs.append(lig)
            rec_paths = [os.path.join(self.dataset_dir, name, f'{name}_protein.pdb') for name in complex_names]
        else:
            for name in tqdm(complex_names, desc='Loading ligands'):
                lig = read_molecule(os.path.join(self.dataset_dir, name, f'{name}_ligand.mol2'), remove_hs=self.remove_h)
                if lig is None:
                    lig = read_molecule(os.path.join(self.dataset_dir, name, f'{name}_ligand.sdf'), remove_hs=self.remove_h)
                ligs.append(lig)
            rec_paths = [os.path.join(self.dataset_dir, name, f'{name}_protein_processed.pdb') for name in complex_names]   

        # Get receptor input
        if not os.path.exists(os.path.join(self.processed_dir, 'rec_input_chain_ids.pt')):
            rec_input = [
                get_receptor_input(r, l, c, cutoff=self.chain_radius)
                for r, l, c in tqdm(
                    zip(rec_paths, ligs, complex_names), desc='Getting receptor input', total=len(complex_names)
                )
            ]
            log('Saving receptor input.')
            torch.save(rec_input, os.path.join(self.processed_dir, 'rec_input_chain_ids.pt'))

        if not self.remove_h and not os.path.exists(os.path.join(self.processed_dir, 'lig_input_framing.pt')):
            self._process_ligands(
                ligs, complex_names, 'lig_input_framing.pt'
            )
        if self.remove_h and not os.path.exists(
            os.path.join(self.processed_dir, 'lig_input_framing_noH.pt')
        ):
            self._process_ligands(
                ligs, complex_names, 'lig_input_framing_noH.pt'
            )

    def _process_chain_ids(self):
        receptors = torch.load(os.path.join(self.processed_dir, 'rec_input_chain_ids.pt'))
        for rec in receptors:
            chain_ids = rec['chain_ids']
            res_per_id = rec['n_res_per_chain']
            total_seq_length = rec['seq_length']
            unique_lengths = list(set(res_per_id.values()))
            chain_ids_processed = torch.empty(total_seq_length)
            entity_ids_processed = torch.empty(total_seq_length)
            sym_ids_processed = torch.empty(total_seq_length)
            start_idx = 0
            existing_ids = {} 
            
            for id, length in res_per_id.items():
                c = torch.tensor([chain_ids.index(id) for _ in range(length)])
                e = torch.tensor([unique_lengths.index(length) for _ in range(length)])
                if length in existing_ids.keys():
                    s = existing_ids[length] + 1
                else:
                    s = 0
                chain_ids_processed[start_idx:start_idx+length] = c
                entity_ids_processed[start_idx:start_idx+length] = e
                sym_ids_processed[start_idx:start_idx+length] = torch.tensor([s for _ in range(length)])
                start_idx += length
                existing_ids[length] = s
        
            rec['chain_ids_processed'] = chain_ids_processed
            rec['entity_ids_processed'] = entity_ids_processed
            rec['sym_ids_processed'] = sym_ids_processed

        torch.save(receptors, os.path.join(self.processed_dir, 'rec_input_proc_ids.pt'))
    
    def _process_unseen(self):
        unseen_pdb_ids = [
            '6qqw', '6jap', '6np2', '6qrc', '6oio', '6jag', '6i9a', '6jb4', '6seo', '6jid', '5ze6', '6pka',
            '6n97', '6qtr', '6n96', '6qzh', '6qqz', '6k3l', '6cjs', '6n9l', '6ott', '6npp', '6nsv', '6n53',
            '6eeb', '6n0m', '6ovz', '5zcu', '6mjq', '6efk', '6gdy', '6kqi', '6ueg', '6qr7', '6g3c', '6iql',
            '6qr4', '6jib', '6qto', '6qrd', '6e5s', '5zlf', '6om4', '6qqv', '6qtq', '6os5', '6s07', '6mjj',
            '6jb0', '6uim', '6mo0', '6cjr', '6uii', '6sen', '6kjf', '6qr9', '6g9f', '6npi', '6oip', '6miv',
            '6qts', '6oi8', '6c85', '6qsz', '6jbb', '6np5', '6nlj', '6n94', '6e13', '6uil', '6n92', '6uhv',
            '6q36', '6qtx', '6rr0', '6ufo', '6oiq', '6qra', '6m7h', '6ufn', '6qr0', '6o5u', '6ny0', '6jan',
            '6ftf', '6jon', '6cf7', '6o9c', '6qqu', '6mja', '6r4k', '6h9v', '6py0', '6jaq', '6k2n', '6cjj',
            '6a73', '6qqt', '6qre', '6qtw', '6np4', '6n55', '6kjd', '6np3', '6jbe', '6qqq', '6j9y', '6h7d',
            '6jao', '6e7m', '6rz6', '6qtm', '6miy', '6jad', '6mj4', '6qr2', '6qxa', '6o9b', '6ckl', '6oir',
            '6oin', '6jam', '6uhu', '6mji', '6nt2', '6op9', '6e4v', '6a87', '6cjp', '6qrf', '6j9w', '6n93',
            '6nd3', '6os6', '6dql', '6qwi', '6npm', '6qrg', '6nxz', '6qr3', '6qr1', '6o5g', '6r7d', '6mo2'
        ]
        all_proteins = torch.load(os.path.join(self.processed_dir, 'rec_input_proc_ids.pt'))
        all_ligands = torch.load(os.path.join(self.processed_dir, 'lig_input_framing.pt'))
        all_ligands_noH = torch.load(os.path.join(self.processed_dir, 'lig_input_framing_noH.pt'))
        unseen_proteins, unseen_ligands, unseen_ligands_noH = [], [], []
        for p, l, l_noH in zip(all_proteins, all_ligands, all_ligands_noH):
            if p['complex_names'] in unseen_pdb_ids:
                unseen_proteins.append(p)
                unseen_ligands.append(l)
                unseen_ligands_noH.append(l_noH)
        torch.save(unseen_proteins, os.path.join(self.processed_dir, 'unseen_rec_input_proc_ids.pt'))
        torch.save(unseen_ligands, os.path.join(self.processed_dir, 'unseen_lig_input_framing.pt'))
        torch.save(unseen_ligands_noH, os.path.join(self.processed_dir, 'unseen_lig_input_framing_noH.pt'))

    def _process_ligands(self, ligs, complex_names, file_name):
        lig_input = [
            process_ligand(l, c, seed=self.seed)
            for l, c in tqdm(
                zip(ligs, complex_names), desc='Getting ligand input', total=len(complex_names)
            )
        ]
        log('Saving ligand input.')
        torch.save(lig_input, os.path.join(self.processed_dir, file_name))