import argparse
import sys
from copy import deepcopy
import os
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm
import torch
import numpy as np
import yaml
import glob
from commons.utils import (
    log, get_parameter_value, rigid_transform_Kabsch_3D, Logger,
    get_symmetry_rmsd, save_predictions_to_file
)
from dataset.process_mols import read_molecule, reorder_atoms
from dataset.dataimporter import DataImporter
from torch.utils.data import DataLoader
from openfold.utils.seed import seed_globally
from quickbind import QuickBind

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--name', type=str)
    p.add_argument('--unseen_only', type=bool, default=False)
    p.add_argument('--save_to_file', type=bool, default=False)
    p.add_argument('--pb_set', type=bool, default=False)
    p.add_argument('--output_s', type=bool, default=False)
    p.add_argument('--val_set', type=bool, default=False)
    p.add_argument('--train_set', type=bool, default=False)
    return p.parse_args()

def collate(data, use_topological_distance, one_hot_adj):
    assert len(data) == 1
    data = data[0]

    aatype = data.aatype.unsqueeze(0).to(device)
    lig_atom_features = data.lig_atom_features.unsqueeze(0).to(dtype=torch.float32).to(device)
    t_true = data.true_lig_atom_coords.unsqueeze(0).to(device)
    rec_mask = torch.ones(data.aatype.shape[0]).unsqueeze(0).to(device)
    lig_mask = torch.ones(data.lig_atom_features.shape[0]).unsqueeze(0).to(device)

    if use_topological_distance:
        adj = torch.clamp(data.distance_matrix.unsqueeze(0), max=7).to(device)
        adj = torch.nn.functional.one_hot(adj.long(), num_classes=8).to(dtype=torch.float32)
    elif one_hot_adj:
        adj = data.adjacency_bo.unsqueeze(0).to(dtype=torch.int64).to(device)
    else:
        adj = data.adjacency.unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32).to(device)
    
    ri = data.residue_index.unsqueeze(0).to(dtype=torch.int64).to(device)
    chain_id = data.chain_ids_processed.unsqueeze(0).to(dtype=torch.int64).to(device)
    entity_id = data.entity_ids_processed.unsqueeze(0).to(dtype=torch.int64).to(device)
    sym_id = data.sym_ids_processed.unsqueeze(0).to(dtype=torch.int64).to(device)
    id_batch = (ri, chain_id, entity_id, sym_id)

    t_rec = data.c_alpha_coords.unsqueeze(0).to(device)
    N = data.n_coords.unsqueeze(0).to(device)
    C = data.c_coords.unsqueeze(0).to(device)
    t_lig = data.lig_atom_coords.unsqueeze(0).to(device)

    pseudo_N = data.pseudo_N.unsqueeze(0).to(device)
    pseudo_C = data.pseudo_C.unsqueeze(0).to(device)

    names = data.complex_name

    return (
        aatype, lig_atom_features, adj, rec_mask, lig_mask, N, t_rec, C, t_lig, id_batch, pseudo_N, pseudo_C
    ), t_true, names


def run_inference(model, test_loader):
    all_ligs_coords_pred, all_ligs_coords, all_masks, all_names = [], [], [], []
    s_pre_struct_lst = []
    for batch, t_true, names in tqdm(test_loader, desc='Generating model predictions'):
        _, _, _, rec_mask, lig_mask, _, _, _, t_lig, _, _, _ = batch
        with torch.no_grad():
            if aux_head or lig_aux_head:
                if args.output_s:
                    out, _, s_pre_struct = model(*batch)
                else:
                    out, _ = model(*batch)
            else:
                if args.output_s:
                    out, s_pre_struct = model(*batch)
                else:
                    out = model(*batch)
            outputs = out[-1][:, rec_mask.shape[-1]:].get_trans()
        all_ligs_coords_pred.append(outputs.detach().cpu())
        all_ligs_coords.append(t_true.detach().cpu())
        all_masks.append(lig_mask.detach().cpu())
        all_names.append(names)
        if args.output_s:
            s_pre_struct_lst.append(s_pre_struct.detach().cpu())
    return {
        'predictions': all_ligs_coords_pred,
        'targets': all_ligs_coords,
        'masks': all_masks,
        'names': all_names,
        's_pre_struct': s_pre_struct_lst
    }

def print_results(rmsds, centroid_distances, incl_H):
    rmsds = np.array(rmsds)
    centroid_distances = np.array(centroid_distances)
    print('----------------------------------------------------------------------------------------------------')
    print(
        f'|                              Test statistics ({"incl." if incl_H else "excl."} hydrogen atoms)                              |'
    )
    print('----------------------------------------------------------------------------------------------------')
    print(f'Mean RMSD: {rmsds.mean().__round__(2)} +- {rmsds.std().__round__(2)}')
    print('RMSD percentiles: ', np.percentile(rmsds, [25, 50, 75]).round(2))
    print(f'% RMSD below 2: {(100 * (rmsds < 2).sum() / len(rmsds)).__round__(2)}%')
    print(f'% RMSD below 5: {(100 * (rmsds < 5).sum() / len(rmsds)).__round__(2)}%')
    print(
        f'Mean centroid distance: {centroid_distances.mean().__round__(2)} +- {centroid_distances.std().__round__(2)}'
    )
    print('Centroid percentiles: ', np.percentile(centroid_distances, [25, 50, 75]).round(2))
    print(
        f'% centroid distances below 2: {(100 * (centroid_distances < 2).sum() / len(centroid_distances)).__round__(2)}%'
    )
    print(
        f'% centroid distances below 5: {(100 * (centroid_distances < 5).sum() / len(centroid_distances)).__round__(2)}%'
    )

def evaluate_predictions(results):
    rmsds_wH, centroid_dists_wH = [], []
    kabsch_rmsds, rmsds, centroid_distances = [], [], []
    for prediction, target, mask, name in tqdm(zip(
        results['predictions'], results['targets'], results['masks'], results['names']),
        desc='Evaluating model predictions', total = len(results['predictions'])
    ):
        if not remove_h:
            # including H atoms
            coords_pred = (prediction * mask.unsqueeze(-1)).numpy()
            coords_native = target.numpy()
            mask = mask.numpy()
            rmsd = np.sqrt(np.sum(np.sum((coords_pred - coords_native) ** 2, axis=1)) / np.sum(mask))
            centroid_distance = np.linalg.norm(
                np.sum(coords_native, axis=0) / np.sum(mask) - np.sum(coords_pred, axis=0) / np.sum(mask)
            )
            centroid_dists_wH.append(centroid_distance)
            rmsds_wH.append(rmsd)

        # not including H atoms
        if args.pb_set:
            lig = read_molecule(os.path.join('data/posebusters_benchmark_set', name, f'{name}_ligand.sdf'), remove_hs=remove_h)
        else:
            lig = read_molecule(os.path.join('data/PDBBind/', name, f'{name}_ligand.mol2'), remove_hs=remove_h)
            if lig is None:
                lig = read_molecule(os.path.join('data/PDBBind/', name, f'{name}_ligand.sdf'), remove_hs=remove_h)
        lig = RemoveHs(lig)
        lig = reorder_atoms(lig)
        lig_pred = deepcopy(lig)
        conf = lig_pred.GetConformer()
        prediction = prediction.squeeze().cpu().numpy()
        for i in range(lig_pred.GetNumAtoms()):
            x, y, z = prediction[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        coords_pred = lig_pred.GetConformer().GetPositions()
        lig_true = deepcopy(lig)
        conf_true = lig_true.GetConformer()
        for i in range(lig_true.GetNumAtoms()):
            x, y, z = target.squeeze()[i]
            conf_true.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        coords_native = lig_true.GetConformer().GetPositions()
        try:
            rmsd = get_symmetry_rmsd(lig_true, coords_native, coords_pred, lig_pred)
        except Exception as e:
            print("Using non corrected RMSD because of the error:", e)
            rmsd = np.sqrt(np.sum((coords_pred - coords_native) ** 2, axis=1).mean())
        centroid_distance = np.linalg.norm(coords_native.mean(axis=0) - coords_pred.mean(axis=0))
        R, t = rigid_transform_Kabsch_3D(coords_pred.T, coords_native.T)
        moved_coords = (R @ (coords_pred).T).T + t.squeeze()
        kabsch_rmsd = np.sqrt(np.sum((moved_coords - coords_native) ** 2, axis=1).mean())
        kabsch_rmsds.append(kabsch_rmsd)
        rmsds.append(rmsd)
        centroid_distances.append(centroid_distance)

    if not remove_h:
        print_results(
            rmsds_wH, centroid_dists_wH, True
        )
    kabsch_rmsds = np.array(kabsch_rmsds)
    print_results(
        rmsds, centroid_distances, False
    )
    print(f'Mean Kabsch RMSD: {kabsch_rmsds.mean().__round__(2)} +- {kabsch_rmsds.std().__round__(2)}')
    print(f'Median Kabsch RMSD: {np.median(kabsch_rmsds).__round__(2)} +- {kabsch_rmsds.std().__round__(2)}')
    print('Kabsch RMSD percentiles: ', np.percentile(kabsch_rmsds, [25, 50, 75]).round(2))

if __name__ == '__main__':
    args = parse_arguments()
    log(f'Loading model {args.name}.')
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f'Using {device}.')
    # Find best model checkpoint
    checkpoints = glob.glob(f'checkpoints/{args.name}/best_checkpoint*.pt')
    best_scores = []
    for checkpoint in checkpoints:
        model_state = torch.load(checkpoint)
        best_scores.append(
            model_state['callbacks'][
                "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
            ]["current_score"]
        )
    best_checkpoint = checkpoints[best_scores.index(min(best_scores))]
    log(f'Model achieved a validation loss of {min(best_scores)}.')
    model_state = torch.load(best_checkpoint)
    state_dict = {
        key[6:]: model_state['state_dict'][key]
        for key in model_state['state_dict']
    }

    with open(f'checkpoints/{args.name}/config.yaml', 'r') as arg_file:
        checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)

    sys.stdout = Logger(logpath=f'checkpoints/{checkpoint_dict["name"]}/inference.log', syspart=sys.stdout)
    sys.stderr = Logger(logpath=f'checkpoints/{checkpoint_dict["name"]}/inference.log', syspart=sys.stderr)
    seed = 0 if checkpoint_dict['seed'] is None else checkpoint_dict['seed']
    seed_globally(seed)
    checkpoint_dict['dataset_params']['cropping'] = False # no cropping at inference time

    if args.pb_set:
        test_data = DataImporter(
            complex_names_path='data/posebusters_benchmark_set/posebusters',
            **checkpoint_dict['dataset_params']
        )
    elif args.val_set:
        # run inference on the validation set
        test_data = DataImporter(complex_names_path=checkpoint_dict['val_names'], **checkpoint_dict['dataset_params'])    
    elif args.train_set:
        # run inference on the training set
        test_data = DataImporter(complex_names_path=checkpoint_dict['train_names'], **checkpoint_dict['dataset_params'])
    else:
        test_data = DataImporter(
            complex_names_path=checkpoint_dict['test_names'],
            **checkpoint_dict['dataset_params'], unseen_only=args.unseen_only
        )
    log(f'Test size: {len(test_data)}.')
    lig_feat_dim, rec_feat_dim = test_data.get_feature_dimensions()
    model = QuickBind(
        aa_feat=rec_feat_dim, lig_atom_feat=lig_feat_dim, **checkpoint_dict['model_parameters'],
        chunk_size=2, output_s=args.output_s
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    aux_head = get_parameter_value('use_aux_head', checkpoint_dict['model_parameters'])
    lig_aux_head = get_parameter_value('use_lig_aux_head', checkpoint_dict['model_parameters'])
    one_hot_adj = get_parameter_value('one_hot_adj', checkpoint_dict['model_parameters'])
    use_topological_distance = get_parameter_value('use_topological_distance', checkpoint_dict['model_parameters'])
    remove_h = get_parameter_value('remove_h', checkpoint_dict['dataset_params'])

    test_loader = DataLoader(
        test_data, batch_size=1, collate_fn=lambda x: collate(
        x, use_topological_distance, one_hot_adj
        )
    )

    out_path = (
        f'checkpoints/{checkpoint_dict["name"]}/'
        f'{"unseen_" if args.unseen_only else ""}'
        f'{"posebusters_" if args.pb_set else ""}'
        f'{"train_" if args.train_set else ""}'
        f'{"val_" if args.val_set else ""}'
        'predictions'
        f'{"-w-single-rep" if args.output_s else ""}.pt'
    )
    if not os.path.exists(out_path):
        results = run_inference(model, test_loader)
        log(f'Saving predictions to {out_path}.')
        torch.save(results, out_path)
    else:
        log('Loading model predictions.')
        results = torch.load(out_path)

    evaluate_predictions(results)

    if args.save_to_file:
        log(f'Saving predictions as SD files.')
        if args.pb_set:
            receptor_dir='data/processed/posebusters'
        else:
            receptor_dir='data/processed/timesplit_test'
        save_predictions_to_file(
            results, receptor_dir,
            os.path.join(f'checkpoints/{args.name}', f'{"posebusters_" if args.pb_set else ""}sdffiles')
        )
