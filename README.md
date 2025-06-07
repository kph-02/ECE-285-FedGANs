# Federated GAN on HAM10000

This project explores training a **Federated GAN (UAGAN)** on a **privacy-sensitive medical imaging dataset (HAM10000)**, providing privacy guarantees while maintaining competitive performance. A **Centralized DCGAN** is also trained for fair comparison using identical generator and discriminator architectures.

---

## Overview

- **Centralized DCGAN** and **Federated UAGAN** are trained on the HAM10000 dataset.
- Both models share the same architecture (based on DCGAN).
- Evaluated on performance and privacy via **Membership Inference Attacks (MIA)**.

---

## Training Setup

- **Dataset**: [HAM10000 - Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **GAN Architecture**: DCGAN-based
- **Optimizer**: Adam
- **Latent Dimension**: 128
- **Batch Size**: 64
- **Epochs**: 200

### Federated Setting (UAGAN)
- **Clients**: 10
- **Local Epochs per Round**: 2
- **Federated Rounds**: 100
- **Aggregation**: Federated Averaging (FedAvg)

---

## Dataset Preparation

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
   Place it under: `UAGAN/datasets/HAM10000/`
2. Merge the images from HAM10000_images_part1 and HAM10000_images_part2 into one single folder images.

## Training Models
1. python train.py --dataroot datasets/HAM10000_processed --name dcgan_ham10000 --model dcgan --dataset_mode ham10000 --batch_size 10
2. Similar command with --model uagan


## Metrics 
1. python metrics_calculation.py --dataroot ./datasets/HAM10000_processed --name uagan_ham10000 --model uagan --epoch 200 --dataset_mode h5
   
