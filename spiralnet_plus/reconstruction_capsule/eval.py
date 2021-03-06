import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh
from data import ComaDataset
from transform import Normalize
from network import AE
from network_eval import eval_error
from dataloader import DataLoader
import utils
import writer
import mesh_sampling

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 32, 32, 64],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, '..', 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset
template_fp = osp.join(args.data_fp, 'template', 'template.ply')
normalize_transform = Normalize()
data_dir = '/storage/Research/Vito/ply_dense_aligned/'
dataset = ComaDataset(data_dir, dtype='train', split='sliced', split_term='sliced', pre_transform=normalize_transform)
dataset_test = ComaDataset(data_dir, dtype='test', split='sliced', split_term='sliced', pre_transform=normalize_transform)

train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=6)

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [4, 4, 4, 4]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

model = AE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list, args.latent_channels, device)

checkpoint_file = '/home/lau/env/code/python/VAE_mesh/spiralnet_plus/reconstruction/out/interpolation_exp/checkpoints/checkpoint_300.pt'
print(checkpoint_file)

checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint['model_state_dict'])  
model.to(device)
print('Number of parameters: {}'.format(utils.count_parameters(model)))
print(model)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


mesh = Mesh(filename=template_fp)
eval_error(model, test_loader, device, dataset_test, args.out_dir, mesh, args.latent_channels)
