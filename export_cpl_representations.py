"""
Export CPL representations from a trained SimplePointNet-CPL checkpoint (CPL demo).
Prefers local data at ./data/modelnet40_normal_resampled/, falls back to ../data/... if missing.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models.simple_pointnet_cpl import build_simple_pointnet_cpl


def parse_args():
    parser = argparse.ArgumentParser('export_cpl_representations_demo')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to best_model.pth')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='dataset split to export')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for export')
    parser.add_argument('--num_point', type=int, default=1024, help='number of input points')
    parser.add_argument('--num_category', type=int, default=40, choices=[10, 40], help='ModelNet10/40')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals if available')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='force cpu')
    parser.add_argument('--out_dir', type=str, default='./exports', help='output directory')
    parser.add_argument('--save_indices', action='store_true', default=True, help='also save CPL indices')
    return parser.parse_args()


def build_loader_from_dat(data_dir: str, num_point: int, batch_size: int, split: str):
    import pickle, numpy as np, torch
    fps_suffix = ''
    dat_name = f"modelnet40_{split}_{num_point}pts{fps_suffix}.dat"
    dat_path = os.path.join(data_dir, dat_name)
    with open(dat_path, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)
    points = torch.from_numpy(np.stack(list_of_points, axis=0))[:, :, 0:3]
    labels = torch.from_numpy(np.stack(list_of_labels, axis=0).squeeze())
    ds = torch.utils.data.TensorDataset(points, labels)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

def main():
    args = parse_args()
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else 'cuda')

    # Data
    data_dir = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')
    loader = build_loader_from_dat(data_dir, args.num_point, args.batch_size, args.split)

    # Model
    model = build_simple_pointnet_cpl(num_points=args.num_point, num_classes=args.num_category, cpl_points=256)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Allocate outputs
    num_samples = len(loader.dataset)
    feats = np.zeros((num_samples, 256, 1024), dtype=np.float32)
    labels = np.zeros((num_samples,), dtype=np.int64)
    idxs = np.zeros((num_samples, 256), dtype=np.int64) if args.save_indices else None

    # Iterate
    offset = 0
    with torch.no_grad():
        for points, target in tqdm(loader, total=len(loader)):
            bsz = points.shape[0]
            points = points.to(device)
            cpl_feats, cpl_idx = model.extract_cpl(points)
            feats[offset:offset+bsz] = cpl_feats.cpu().numpy()
            labels[offset:offset+bsz] = target.numpy()
            if idxs is not None:
                idxs[offset:offset+bsz] = cpl_idx.cpu().numpy()
            offset += bsz

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir.joinpath(f'cpl_features_{args.split}.npy')), feats)
    np.save(str(out_dir.joinpath(f'labels_{args.split}.npy')), labels)
    if idxs is not None:
        np.save(str(out_dir.joinpath(f'cpl_indices_{args.split}.npy')), idxs)

    print('Export complete:')
    print(f'  features -> {out_dir}/cpl_features_{args.split}.npy')
    print(f'  labels   -> {out_dir}/labels_{args.split}.npy')
    if idxs is not None:
        print(f'  indices  -> {out_dir}/cpl_indices_{args.split}.npy')


if __name__ == '__main__':
    main()


