# Recursive Reasoning in Minimax Games: A Level  Gradient Play Method

This repository is the official implementation of [Recursive Reasoning in Minimax Games: A Level  Gradient Play Method]. 

## Requirements

Requirements:

python3
pytorch
torchvision
numpy
scipy

All codes are developed using Python 3.9, pytorch 1.9.0 and CUDA Version: 11.4 

## Training

To train the model(s) in the paper on CIFAR-10 dataset, you should download the Precalculated Statistics [fid_stats_cifar10_train.npz](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) to `./SNGAN`.

After downloading the precalcuated stats, run this command:

```train
python main.py
```


## Evaluation

To evaluate our model using the official Tensorlow implementation of FID score, you should follow the instruction in the colab notebook.

## Pre-trained Models

We provided a best performing generator trained on CIFAR-10 dataset with which we obtained a FID score of 10.12. You can use the colab notebook to evaluate its performance.

## Results

Our model achieves a FID score of 10.12 on CIFAR-10 dataset and 25.43 on STL-10 dataset. 
https://paperswithcode.com/sota/image-generation-on-cifar-10
https://paperswithcode.com/sota/image-generation-on-stl-10

