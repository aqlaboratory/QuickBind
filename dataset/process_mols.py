import warnings
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Polypeptide import protein_letters_3to1
from rdkit import Chem
from rdkit.Chem import AllChem, GetDistanceMatrix 
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdchem import Conformer
from rdkit.Geometry import Point3D
from rdkit import RDLogger
from scipy import spatial
from openfold.np import residue_constants
from itertools import chain
from typing import Optional

def read_molecule(file: str, sanitize: Optional[bool] = True, remove_hs: Optional[bool] = False):
    """Reads in ligand from an input file.
    Args:
      file : Path to ``.mol2``, ``.sdf`` or ``.pdb`` input file.
      sanitize : Whether to sanitizate the molecule.
      remove_hs : Whether to remove hydrogens.
    Returns:
      Ligand parsed from the input file.
    """
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*') 
    if file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(file, sanitize=False, removeHs=False)
    elif file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Unsupported file format.')

    try:
        if sanitize:
            Chem.SanitizeMol(mol)
        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
        return mol
    except Exception:
        return None

# LIGAND PROCESSING

lig_features = {
    'atomic_number': [1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 'misc'],
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'degree': [1, 2, 3, 4, 'misc'],
    'formal_charge': [-1, 0, 1, 'misc'],
    'numH': [0, 1, 2, 3, 'misc'],
    'hybridisation': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True]
}

def reorder_atoms(ligand, scheme = 'canonicalatomrank'):
    """
    Reorders ligand atoms based on global molecular information.
    Args: 
        ligand (rdkit.Chem.rdchem.Mol object):
            Input ligand
        scheme (str):
            Scheme to use. Valid options are 'canonicalatomrank'
            and 'longestlinearchain'.
            Default: 'canonicalatomrank'.
    Returns: 
        reordered_ligand (rdkit.Chem.rdchem.Mol object):
            Ligand with reordered atoms. 
    """
    if scheme == 'canonicalatomrank': 
        neworder = list(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(ligand))])))[1]
    elif scheme == 'longestlinearchain': 
        smiles = Chem.MolToSmiles(ligand)
        neworder = list(map(int, ligand.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    else:
        raise ValueError("Invalid renumbering scheme.")
    reordered_ligand = Chem.RenumberAtoms(ligand, neworder)

    conf = ligand.GetConformer()
    reordered_conf = reordered_ligand.GetConformer()
    for i in range(ligand.GetNumAtoms()):
        x, y, z = Conformer.GetPositions(conf)[i]
        reordered_conf.SetAtomPosition(neworder.index(i), Point3D(float(x), float(y), float(z))) 
    
    Chem.SanitizeMol(reordered_ligand)
    return reordered_ligand

def process_ligand(mol, name, seed=None):
    mol = reorder_atoms(mol)
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    adjacency = GetAdjacencyMatrix(mol)
    adjacency_bo = GetAdjacencyMatrix(mol, useBO=True)
    distance_matrix = GetDistanceMatrix(mol)
    neighbour_list = get_neighbour_list(adjacency)
    try:
        lig_coords = get_rdkit_coords(mol, seed).numpy()
    except Exception as e:
        lig_coords = true_lig_coords
        with open('ligand_processing.log', 'a') as f:
            f.write(f'Generating RDKit conformer failed for\n{name}\n{str(e)}\n')
            f.flush()
        print(f'Generating RDKit conformer failed for {name}, {e}')
    pseudo_N, pseudo_C, true_pseudo_N, true_pseudo_C = get_adjacent_atoms(neighbour_list, lig_coords, true_lig_coords)

    features = {"atom_features": get_lig_atom_features(mol)}
    features["atom_coords"] = torch.tensor(lig_coords, dtype=torch.float32)
    features["true_atom_coords"] = torch.tensor(true_lig_coords, dtype=torch.float32)
    features["adjacency"] = torch.tensor(adjacency, dtype=torch.int)
    features["adjacency_bo"] = torch.tensor(adjacency_bo, dtype=torch.int)
    features["distance_matrix"] = torch.tensor(distance_matrix, dtype=torch.int)
    features["pseudo_N"] = pseudo_N.to(dtype=torch.float32)
    features["pseudo_C"] = pseudo_C.to(dtype=torch.float32)
    features["true_pseudo_N"] = true_pseudo_N.to(dtype=torch.float32)
    features["true_pseudo_C"] = true_pseudo_C.to(dtype=torch.float32)

    return features

def get_rdkit_coords(mol, seed=None):
    ETKDG = AllChem.ETKDGv2()
    if seed is not None:
        ETKDG.randomSeed = seed
    new_conf_id = AllChem.EmbedMolecule(mol, ETKDG)
    if new_conf_id == -1:
        ETKDG.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ETKDG)
    AllChem.MMFFOptimizeMolecule(mol, confId=0)
    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return torch.tensor(lig_coords, dtype=torch.float32)

def get_lig_atom_features(mol):
    return torch.tensor(
        [
        [
            get_index(
                lig_features['atomic_number'], atom.GetAtomicNum()
            ),
            lig_features['chirality'].index(
                str(atom.GetChiralTag())
            ),
            get_index(
                lig_features['degree'], atom.GetTotalDegree()
            ),
            get_index(
                lig_features['formal_charge'],
                atom.GetFormalCharge(),
            ),
            get_index(
                lig_features['numH'], atom.GetTotalNumHs()
            ),
            get_index(
                lig_features['hybridisation'],
                str(atom.GetHybridization()),
            ),
            lig_features['is_aromatic'].index(
                atom.GetIsAromatic()
            ),
            get_index(
                lig_features['is_in_ring'],
                atom.IsInRing(),
            ),
        ]
        for atom in mol.GetAtoms()
        ]
    )

def get_index(l, e):
    try:
        return l.index(e)
    except Exception:
        return len(l) - 1
    
def get_neighbour_list(adjacency):
    neighbour_list = []
    for row in adjacency:
        neighbours = [neighbour for neighbour, entry in enumerate(row) if entry == 1]
        neighbour_list.append(neighbours)
    return neighbour_list

def get_adjacent_atoms(neighbour_list, lig_coords, true_lig_coords):
    """ 
    For each ligand atom, get the coordinates of the two adjacent ligand atoms with the 
    lowest indices. These will be used as pseudo N and C atoms, analogously to the N and 
    C atoms for the protein.
    
    Procedure:
    (1) If a given atom has two or more bonds, then choose the two neighbours two with the lowest indices.
        If the ligand atoms were reordered using the canonical atom ranking, these would be the two with
        the lowest priority.
    (2) If it has only one bond, obtain the coordinates of a dummy atom as follows:
        (1) Compute the bond vector of the single bond ("x_y").
        (2) Copy that vector ("x_z").
        (3) While keeping the x and y coordinates constant, find the z coord such that the dot product of the two bond vectors x_y and x_z is 0. 
        (4) Subtract the new x_z bond vector from the atom coordinate to get the coordinates of the dummy atom.
    """

    pseudo_N, pseudo_C = torch.empty(0, 3), torch.empty(0, 3)
    true_pseudo_N, true_pseudo_C = torch.empty(0, 3), torch.empty(0, 3)

    for i in range(len(neighbour_list)):
        pseudo_N = torch.cat(
            (pseudo_N, torch.tensor(lig_coords[neighbour_list[i][0]]).unsqueeze(0)), 0
        )
        true_pseudo_N = torch.cat(
            (true_pseudo_N, torch.tensor(true_lig_coords[neighbour_list[i][0]]).unsqueeze(0)), 0
        )
        if len(neighbour_list[i]) >= 2:
            pseudo_C = torch.cat(
                (pseudo_C, torch.tensor(lig_coords[neighbour_list[i][1]]).unsqueeze(0)), 0
            )
            true_pseudo_C = torch.cat(
                (true_pseudo_C, torch.tensor(true_lig_coords[neighbour_list[i][1]]).unsqueeze(0)), 0
            )
        else:
            origin = lig_coords[i]
            N = lig_coords[neighbour_list[i][0]]
            x_y = origin - N

            if abs(x_y[-1]) <= 0.0001:
                x_y[-1] = 0.0001 if x_y[-1] > 0 else -0.0001
            x_z = x_y.copy()

            x_z[-1] = -(x_z[:-1] @ x_y[:-1]) / x_y[-1]
            dummy_coord = torch.tensor(origin - x_z)
            pseudo_C = torch.cat((pseudo_C, dummy_coord.unsqueeze(0)), 0)

            origin_t = true_lig_coords[i]
            N_t = true_lig_coords[neighbour_list[i][0]]
            x_y_t = origin_t - N_t

            if abs(x_y_t[-1]) <= 0.0001:
                x_y_t[-1] = 0.0001 if x_y_t[-1] > 0 else -0.0001
            x_z_t = x_y_t.copy()

            x_z_t[-1] = -(x_z_t[:-1] @ x_y_t[:-1]) / x_y_t[-1]
            dummy_coord_t = torch.tensor(origin_t - x_z_t)
            true_pseudo_C = torch.cat((true_pseudo_C, dummy_coord_t.unsqueeze(0)), 0)

    return pseudo_N, pseudo_C, true_pseudo_N, true_pseudo_C

# RECEPTOR PROCESSING

def get_receptor_input(rec_path, lig, complex_name, cutoff):
    biopython_parser = PDBParser()
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]

    c_alpha_coords, n_coords, c_coords, three_letter_sequence, n_res_per_chain, valid_chain_ids = get_rec_data(rec, lig_coords, cutoff)

    one_letter_sequence = convert_sequence(three_letter_sequence)
    num_res = len(one_letter_sequence)

    features = {
        "c_alpha_coords": c_alpha_coords,
        "n_coords": n_coords,
        "c_coords": c_coords,
        "aatype": residue_constants.sequence_to_onehot(
            sequence=one_letter_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        ).astype(np.float32),
        "residue_index": np.array(range(num_res), dtype=np.int32),
        "seq_length": num_res
    }

    tensor_dict = {
        k: torch.tensor(v) for k, v in features.items()
    }
    
    tensor_dict["complex_names"] = complex_name
    tensor_dict["chain_ids"] = valid_chain_ids
    tensor_dict["n_res_per_chain"] = n_res_per_chain
    sequence = ''.join(one_letter_sequence)
    tensor_dict["sequence"] = sequence

    return tensor_dict

def get_rec_data(receptor, lig_coords, cutoff):
    min_distances, c_alpha_coords, n_coords, \
        c_coords, valid_chain_ids, sequence  = [], [], [], [], [], []
    
    for polypep in receptor:
        chain_coords, chain_c_alpha_coords, chain_n_coords, \
            chain_c_coords, invalid_res_ids, chain_sequence  = [], [], [], [], [], []
        
        for residue in polypep:
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []

            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))

            if c_alpha is None or n is None or c is None:
                invalid_res_ids.append(residue.get_id())
            else: 
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                chain_sequence.append(residue.get_resname())

        for res_id in invalid_res_ids:
            polypep.detach_child(res_id)

        if chain_coords:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf

        min_distances.append(min_distance)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        sequence.append(chain_sequence)
        if min_distance < cutoff:
            valid_chain_ids.append(polypep.get_id())

    if not valid_chain_ids:
        valid_chain_ids.append(np.argmin(np.array(min_distances)))

    valid_c_alpha_coords, valid_n_coords, \
        valid_c_coords, valid_sequence = [], [], [], []
    for i, polypep in enumerate(receptor):
        if polypep.get_id() in valid_chain_ids:
            valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_sequence.append(sequence[i])

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    valid_sequence = list(chain.from_iterable(valid_sequence))

    n_res_per_chain = {
        id: len(s) for id, s in zip(valid_chain_ids, valid_sequence)
    }

    return c_alpha_coords, n_coords, c_coords, valid_sequence, n_res_per_chain, valid_chain_ids

def convert_sequence(three_letter_sequence):
    sequence = []
    for res in three_letter_sequence:
        try:
            one_letter_code = protein_letters_3to1[res]
        except KeyError:
            one_letter_code = 'X'
        sequence.append(one_letter_code)
    return sequence
