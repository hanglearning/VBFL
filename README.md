# Documentation under construction...
# VBFL (emulation)

## Introduction
VBFL is a Proof-of-Stake (PoS) blockchain-based Federated Learning framework with a validation scheme robust against the distorted local model updates. This repo hosts an emulated implementation for VBFL written in Python.

Please refer to [*Robust Blockchained Federated Learning with Model Validation and Proof-of-Stake Inspired Consensus*](https://arxiv.org/abs/2101.03300) for detailed explanations of the mechanisms.

## Instructions to Run
**<ins>Suggested</ins> Environments Setup**
```
python 3.7.6
pytorch 1.4.0
```
(1) Clone the repo
```
$ git clone https://github.com/hanglearning/VBFL.git
```
(2) Create a new conda environment with python 3.7.6
```
$ conda create -n VBFL python=3.7.6
$ conda activate VBFL
```
(3) Head to https://pytorch.org/ for instructions to install pytorch
```
$ # If you use CUDA, check its version first
$ nvidia-smi
$ # Install the correct CUDA version for pytorch. 
$ # The code was tested on CUDA 10.1 and pytorch 1.4.0
$ conda install pytorch=1.4.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
(4) Install pycryptodome 3.9.9
```
$ conda install pycryptodome=3.9.9
```
**Run VBFL Simulation**

(1) Create a <i>logs/</i> folder in the VBFL root folder if not exists. Notice that <i>logs/</i> is speicifed in <i>.gitignore</i>.
```
$ cd VBFL
$ mkdir logs/
```
## Acknowledgments

(1)The code of the blockchain architecture and PoW consensus is inspired by Satwik's [*python_blockchain_app*](https://github.com/satwikkansal/python_blockchain_app). 

(2)The code of FedAvg used in VBFL is inspired by [*WHDY's FedAvg implementation*](https://github.com/WHDY/FedAvg).