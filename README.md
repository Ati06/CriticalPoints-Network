## CPLdemo

**A PyTorch implementation of the CPL paper (CVPR 2020) with a PointNet baseline for ModelNet40 classification.**

## How to run
This project has been tested with:
Python - 3.9
PyTorch - 2.4.0
CUDA - 12.1

- Install dependencies (install PyTorch appropriate for your system first), then:
```bash
pip install -r requirements.txt
```

- Train (saves best checkpoint to `log/classification/<timestamp>/checkpoints/best_model.pth`):
```bash
python train_simple_pointnet_cpl.py --gpu 0 --batch_size 24 --epoch 200 --num_point 1024 --num_category 40
# CPU: add --use_cpu
```

- Export CPL representations for a split (writes `.npy` files to `data_utils/exports/`):
```bash
python export_cpl_representations.py --checkpoint log/classification/<timestamp>/checkpoints/best_model.pth --split test --num_point 1024 --num_category 40 --out_dir data_utils/exports
# CPU: add --use_cpu
```