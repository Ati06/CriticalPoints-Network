"""
Train SimplePointNet-CPL on ModelNet40 and save best checkpoint (CPL demo).
Prefers local data at ./data/modelnet40_normal_resampled/, falls back to ../data/... if missing.
"""

import os
import sys
import argparse
import datetime
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import provider
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from models.simple_pointnet_cpl import build_simple_pointnet_cpl


def parse_args():
    parser = argparse.ArgumentParser('train_simple_pointnet_cpl_demo')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='ModelNet10/40')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='preprocess dataset to .dat')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use FPS uniform sampling in preprocessing')
    parser.add_argument('--step_size', type=int, default=20, help='LR step size')
    parser.add_argument('--gamma', type=float, default=0.7, help='LR decay gamma')
    return parser.parse_args()


def build_loader_from_dat(args, data_path: str, split: str):
    fps_suffix = '_fps' if args.use_uniform_sample else ''
    dat_name = f"modelnet{args.num_category}_{split}_{args.num_point}pts{fps_suffix}.dat"
    dat_path = os.path.join(data_path, dat_name)
    if not os.path.exists(dat_path):
        return None
    with open(dat_path, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)
    points = torch.from_numpy(np.stack(list_of_points, axis=0))
    labels = torch.from_numpy(np.stack(list_of_labels, axis=0).squeeze())
    if not args.use_normals:
        points = points[:, :, 0:3]
    ds = torch.utils.data.TensorDataset(points, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=(split=='train'), num_workers=0, drop_last=(split=='train'))
    return loader


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for points, target in tqdm(loader, total=len(loader)):
        points = points.to(device)
        target = target.to(device)
        logits = model(points)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.numel()
    return correct / max(total, 1)


def _resolve_data_path():
    candidates = [
        os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled'),
        os.path.join(BASE_DIR, '..', 'data', 'modelnet40_normal_resampled'),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return os.path.normpath(p) + os.sep
    # Default to local
    return os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled') + os.sep


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else 'cuda')

    # Directories
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/').joinpath('classification')
    exp_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    # Logger
    logger = logging.getLogger('SimplePointNetCPL')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_dir.joinpath('simple_pointnet_cpl.txt')))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.info(args)

    # Data
    data_path = _resolve_data_path()
    try:
        train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
        test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    except FileNotFoundError:
        logger.info('Meta files missing; using .dat preprocessed loaders if available.')
        train_loader = build_loader_from_dat(args, data_path, 'train')
        test_loader = build_loader_from_dat(args, data_path, 'test')
        if train_loader is None or test_loader is None:
            raise

    # Model, loss, optim
    model = build_simple_pointnet_cpl(num_points=args.num_point, num_classes=args.num_category, cpl_points=256)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Train
    best_acc = 0.0
    best_epoch = 0
    logger.info('Start training...')
    for epoch in range(args.epoch):
        model.train()
        running_correct = 0
        running_total = 0
        for points, target in tqdm(train_loader, total=len(train_loader)):
            points_np = points.numpy()
            points_np = provider.random_point_dropout(points_np)
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
            points = torch.from_numpy(points_np)

            points = points.to(device)
            target = target.to(device).long()

            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            running_correct += (pred == target).sum().item()
            running_total += target.numel()

        train_acc = running_correct / max(running_total, 1)
        scheduler.step()

        with torch.no_grad():
            test_acc = evaluate(model, test_loader, device)

        logger.info(f"Epoch {epoch+1}/{args.epoch} - train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            savepath = str(checkpoints_dir.joinpath('best_model.pth'))
            state = {
                'epoch': best_epoch,
                'test_acc': best_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }
            torch.save(state, savepath)
            logger.info(f"Saved best checkpoint to {savepath}")

    logger.info('Training done.')


if __name__ == '__main__':
    main()


