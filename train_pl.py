import argparse
import yaml
import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from commons.utils import log, get_parameter_value
from dataset.dataimporter import DataImporter
from openfold.utils.seed import seed_globally
from quickbind import QuickBind_PL
from openfold.utils.rigid_utils import Rigid, Rotation
torch.cuda.empty_cache()

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'))
    p.add_argument('--resume', type=bool, default=False)
    p.add_argument('--id', type=str, default=None, help='W&B ID')
    p.add_argument('--finetune', type=bool, default=False)
    return p.parse_args()

def collate(data, construct_frames, use_topological_distance, one_hot_adj):
    assert len(data) == 1
    data = data[0]
    aatype = data.aatype.unsqueeze(0)
    lig_atom_features = data.lig_atom_features.unsqueeze(0).to(dtype=torch.float32)
    t_true = data.true_lig_atom_coords.unsqueeze(0)
    rec_mask = torch.ones(data.aatype.shape[0]).unsqueeze(0)
    lig_mask = torch.ones(data.lig_atom_features.shape[0]).unsqueeze(0)

    if use_topological_distance:
        adj = torch.clamp(data.distance_matrix.unsqueeze(0), max=7)
        adj = torch.nn.functional.one_hot(adj.long(), num_classes=8).to(dtype=torch.float32)
    elif one_hot_adj:
        adj = data.adjacency_bo.unsqueeze(0).to(dtype=torch.int64)
    else:
        adj = data.adjacency.unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32)
    
    ri = data.residue_index.unsqueeze(0).to(dtype=torch.int64)
    chain_id = data.chain_ids_processed.unsqueeze(0).to(dtype=torch.int64)
    entity_id = data.entity_ids_processed.unsqueeze(0).to(dtype=torch.int64)
    sym_id = data.sym_ids_processed.unsqueeze(0).to(dtype=torch.int64)
    id_batch = (ri, chain_id, entity_id, sym_id)

    t_rec = data.c_alpha_coords.unsqueeze(0)
    N = data.n_coords.unsqueeze(0)
    C = data.c_coords.unsqueeze(0)
    t_lig = data.lig_atom_coords.unsqueeze(0)

    pseudo_N = data.pseudo_N.unsqueeze(0)
    pseudo_C = data.pseudo_C.unsqueeze(0)
    true_pseudo_N = data.true_pseudo_N.unsqueeze(0)
    true_pseudo_C = data.true_pseudo_C.unsqueeze(0)

    if construct_frames:
        t_true = Rigid.from_3_points(true_pseudo_N, t_true, true_pseudo_C)
    else:
        t_true = Rigid(
                        rots = Rotation.identity(
                            shape=t_true.shape[:-1], dtype = torch.float32, fmt="quat"
                        ), trans = t_true
        )

    return (aatype, lig_atom_features, adj, rec_mask, lig_mask, N, t_rec, C, t_lig, id_batch, pseudo_N, pseudo_C), t_true

def train(config):
    seed = 0 if config['seed'] is None else config['seed']
    seed_globally(seed)
    pl.seed_everything(seed, workers=True)

    if args.finetune:
        config['dataset_params']['crop_size'] = 512

    log('Getting training data.')
    train_data = DataImporter(complex_names_path=config['train_names'], **config['dataset_params'])
    log('Getting validation data.')
    val_data = DataImporter(complex_names_path=config['val_names'], **config['dataset_params'])

    lig_feat_dim, rec_feat_dim = train_data.get_feature_dimensions()
    one_hot_adj = get_parameter_value('one_hot_adj', config['model_parameters'])
    use_topological_distance = get_parameter_value('use_topological_distance', config['model_parameters'])
    construct_frames = get_parameter_value('construct_frames', config['model_parameters'])

    train_loader = DataLoader(
        train_data, batch_size=config['batch_size'], shuffle=True,
        collate_fn=lambda x: collate(x, construct_frames, use_topological_distance, one_hot_adj),
        num_workers=config['num_workers'], prefetch_factor=12
    )
    val_loader = DataLoader(
        val_data, batch_size=config['batch_size'],
        collate_fn=lambda x: collate(x, construct_frames, use_topological_distance, one_hot_adj),
        num_workers=config['num_workers']
    )

    model = QuickBind_PL(
        aa_feat=rec_feat_dim, lig_atom_feat=lig_feat_dim, 
        **config['model_parameters'], loss_config=config['loss_params'], **config['optimizer_params'],
        chunk_size=None
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{config["name"]}',
        filename='best_checkpoint',
        monitor='val_loss',
        save_top_k=3,
        mode="min",
    )
    checkpoint_callback.FILE_EXTENSION = ".pt"
    lr_monitor = LearningRateMonitor(logging_interval='step')
    clip_grad = config['clip_grad'] or 0.0
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        logger=wandb_logger,
        accumulate_grad_batches=config['iters_to_accumulate'], # Gradient accumulation
        precision=16, # Mixed precision training
        gradient_clip_val=clip_grad,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        default_root_dir=f'checkpoints/{config["name"]}',
        deterministic=True, accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        num_nodes=2
    )
    if args.resume:
        checkpoints = glob.glob(f'checkpoints/{config["name"]}/best_checkpoint*.pt')
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_checkpoint)
    elif args.finetune:
        model_state = torch.load(f'checkpoints/{config["name"]}/best_checkpoint.pt')
        model.load_state_dict(model_state['state_dict'])
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    args = parse_arguments()
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    config = yaml.load(args.config, Loader=yaml.FullLoader)
    if args.resume:
        assert args.id is not None, log('If you want to resume an experiment, you should provide the W&B ID!')
        wandb_logger = WandbLogger(name=config['name'], project=config['wandb']['project'], id=args.id)
    else:
        wandb_logger = WandbLogger(name=config['name'], project=config['wandb']['project'])
    name = config['name']
    if not os.path.exists(f'checkpoints/{name}'):
        os.mkdir(f'checkpoints/{name}')
    train(config) 
