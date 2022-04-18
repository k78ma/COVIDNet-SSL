# Training/Evaluation/Inference with COVIDNet-SSL

## Introduction
Released COVIDNet-SSL models are based on ResNet-50 backbone with self-supervised pre-training on the SimCLR framework. Input shape is (3, 244, 244). Below are simple instructions for training, evaluation and validation using these models. 

## Requirements
- PyTorch 1.8.0
- Python 3.6
- Numpy
- Scikit-Learn
- Matplotlib
- TensorBoard Logger

### Directly installing conda environment
To install the conda enviroment on Windows, run
```
conda env create -f /path/to/win_env.yml 
```
To install the conda enviroment on Linux, run
```
conda env create -f /path/to/lin_env.yml
```
To activate the conda enviroment, run
```
conda activate covid-net
```
## Steps for training
We provide you with the PyTorch training script, train.py. This script will train starting from a specified checkpoint; after training, it will save five checkpoints: best validation accuracy, best AUC, best F1, lowest validation loss, and last epoch. It will also create training logs and various diagrams such as validation loss curves and learning rate graphs.
To train:
1. Locate the checkpoint files (location of pretrained model)
2. For basic training from the COVIDNet-SSL pretrained model:
```
python train.py \
  --ckpt-path 'checkpoints/COVIDNet-SSL.ckpt' \
  --data-dir '../archive' \
  --exp-dir './train_folder' \
  --optim 'SGD'
```
3. For more options and information:
```
python train.py --help
```

## Steps for evaluation
We provide you with the PyTorch evaluation script, eval.py. This script will evaluate a given checkpoint; it will calculate top-1 accuracy, ROC-AUC, trust score, as well as precision, recall and F1 for each class. 
To evaluate:e
1. Locate the checkpoint files
2. To evaluate a checkpoint for COVIDNet-SSL:
```
python train.py \
  --ckpt-path 'checkpoints/COVIDNet-SSL.ckpt' \
  --data-dir '../archive' \
  --exp-dir './train_folder'
```
3. For more options and information:
```
python eval.py --help
```

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**
1. Download a model from the pre-trained models section
2. Locate model checkpoints and chest X-Ray image to be inferenced
3. To use inference script for COVIDNet-SSL:
```
python inferene.py \
  --ckpt-path 'checkpoints/COVIDNet-SSL.ckpt' \
  --img-path '../archive/test/MIDRC-RICORD-1C-419639-002463-12463-0.png' \
  --exp-dir './inference_folder'
```
4. For more options and information:
```
python inference.py --help
```

