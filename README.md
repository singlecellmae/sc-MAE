# Single-cell Masked Autoencoder for Clustering and Perturbation of scRNA-seq

This repository contains the pytorch implementation of contrastive-sc, and will include our proposed sc-MAE in the final report.

## Environment

To setup the environment, please simply run

```
pip install -r requirements.txt
```

## Datasets

###  scRNA-seq 

Data can be downloaded from [here](https://drive.google.com/file/d/1JKcLwZypAk8JIn44jt8DVj3wpU_ag2Wd/view?usp=sharing)

**Notice:** please update the path of data (--path) in main.py accordingly.



## Train & Test

For training and testing on 10X_PBMC dataset, please run

```
python main.py --dataset 10X_PBMC
```
