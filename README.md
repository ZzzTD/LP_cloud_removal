# LP_cloud_removal
This is the code implementation of "Thin Cloud Removal Method Based on Low-Frequency Residual Diffusion and High-Frequency Modulation Refinement by Laplacian Pyramid Decoupling"
The paper is currently undergoing external review, and the code will be continuously improved in the future.

## Requirements
1. Install Stage 1 environment, refer to [RDDM](https://github.com/nachifur/RDDM)
2. Install Stage 2 environment, refer to [Restormer](https://github.com/swz30/Restormer)

## Datasets
1. RICE [paper](https://arxiv.org/abs/1901.00600)  

2. T-Cloud [paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Ding_Uncertainty-Based_Thin_Cloud_Removal_Network_via_Conditional_Variational_Autoencoders_ACCV_2022_paper.pdf)

3. CUHK-CR [paper](https://ieeexplore.ieee.org/abstract/document/10552304/)

## Train and Test
1. Stage1 train
```train
cd Stage1
python train.py
```

2. Stage2 train
Before training,place the first stage weights on line 467 of ./Stage2/basicsr/models/archs
```train
cd ./Stage2/basicsr
python train.py -opt= ./Stage2/Stage2_cloud_removal.yml
```

3. test
```test
cd ./Stage2
python test.py -opt= ./Stage2/Stage2_cloud_removal.yml
```

### Acknowledgments
Our code is developed based on [RDDM](https://github.com/nachifur/RDDM) and [Restormer](https://github.com/swz30/Restormer).Thank them for their outstanding work.
