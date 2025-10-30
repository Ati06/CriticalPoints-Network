## Critical Points Network Implementation using PointNet backbone

**A PyTorch implementation of the CPL paper (CVPR 2020) with a PointNet baseline for ModelNet40 classification.**

## How to run
This project has been tested on :
```bash
Python - 3.9
PyTorch - 2.4.0
CUDA - 12.1
```
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

Data is expected in data/modelnet40_normal_resampled/.

data/modelnet40_normal_resampled/
  - modelnet40_shape_names.txt
  - modelnet40_train_1024pts.dat
  - modelnet40_test_1024pts.dat

Processed data can be downloaded from [here](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Reference
[Adaptive Hierarchical Down-Sampling for Point Cloud Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Nezhadarya_Adaptive_Hierarchical_Down-Sampling_for_Point_Cloud_Classification_CVPR_2020_paper.pdf) <br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)<br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
