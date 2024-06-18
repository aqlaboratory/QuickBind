from datetime import datetime
import math
import numpy as np
from spyrmsd import rmsd, molecule
import sys
import torch
import os
from dataset.process_mols import read_molecule, reorder_atoms
from rdkit.Chem import RemoveHs, SDWriter
from rdkit.Geometry import Point3D

def log(*args):
    print(f'[{datetime.now()}]', *args)

def get_parameter_value(param, config):
    try:
        param_value = bool(config[param])
    except Exception:
        param_value = False
    return param_value

class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def save_predictions_to_file(results, receptor_dir, out_path):
    receptors = torch.load(os.path.join(receptor_dir, "rec_input_proc_ids.pt"))
    coms = []
    for rec in receptors:
        c_alpha_coords = rec['c_alpha_coords']
        if c_alpha_coords.shape[0] > 2000: # inference is currently limited to complexes with less than 2000 residues
            continue
        c = torch.mean(c_alpha_coords, dim=0)
        coms.append(c)

    pred_mols = []
    for prediction, name, com in zip(results['predictions'], results['names'], coms):
        if os.path.basename(receptor_dir) == 'posebusters':
            lig = read_molecule(os.path.join('data/posebusters_benchmark_set', name, f'{name}_ligand.sdf'), remove_hs=True)
        else:
            lig = read_molecule(os.path.join('data/PDBBind/', name, f'{name}_ligand.mol2'), remove_hs=True)
            if lig is None:
                lig = read_molecule(os.path.join('data/PDBBind/', name, f'{name}_ligand.sdf'), remove_hs=True)
        lig = RemoveHs(lig)
        lig = reorder_atoms(lig)
        conf = lig.GetConformer()
        p = prediction.squeeze().numpy()
        c = com.numpy()
        for i in range(lig.GetNumAtoms()):
            x, y, z = p[i] + c
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))    
        pred_mols.append(lig)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    for mol, name in zip(pred_mols,results['names']):
        with SDWriter(os.path.join(out_path, f'{name}_pred.sdf')) as w:
            w.write(mol)

def read_strings_from_txt(path):
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]

# This function is taken from EquiBind
# Copyright (c) 2022 Hannes Stärk
# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

# This function is taken from DiffDock
# Copyright (c) 2022 Gabriele Corso, Hannes Stärk, Bowen Jing
def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    mol = molecule.Molecule.from_rdkit(mol)
    mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
    mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
    mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
    RMSD = rmsd.symmrmsd(
        coords1,
        coords2,
        mol.atomicnums,
        mol2_atomicnums,
        mol.adjacency_matrix,
        mol2_adjacency_matrix,
    )
    return RMSD
