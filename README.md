# Training, Evaluation and Inference with COVIDNet-SSL

## Introduction
Released COVIDNet-SSL models are based on ResNet-50 architecture with self-supervised pre-training on the SimCLR framework. Input shape is (3, 244, 244). Below are simple instructions for training, evaluation and validation using these models. 

## Requirements
For now, I just have everything that came up when I ran "conda list".
```
_libgcc_mutex             0.1                        
_openmp_mutex             4.5       
absl-py                   0.14.1                   
astunparse                1.6.3                    
backcall                  0.2.0                      
blas                      1.0                         
blessings                 1.7                     
bzip2                     1.0.8                
ca-certificates           2020.10.14                  
cached-property           1.5.2                 
cachetools                4.2.4                 
captum                    0.4.0                   
certifi                   2020.6.20               
charset-normalizer        2.0.6                   
clang                     5.0                      
colorama                  0.4.4              
cudatoolkit               11.1.1             
cycler                    0.10.0                  
dataclasses               0.8                
decorator                 4.4.2                 
ffmpeg                    4.3                 
flatbuffers               1.12                    
freetype                  2.10.4              
gast                      0.4.0                 
giflib                    5.2.1              
gmp                       6.2.1                
gnutls                    3.6.13              
google-auth               1.35.0                  
google-auth-oauthlib      0.4.6                   
google-pasta              0.2.0                  
gpustat                   0.6.0                    
grpcio                    1.41.0                   
h5py                      3.1.0                 
idna                      3.2                    
imageio                   2.9.0                 
importlib-metadata        4.8.1               
intel-openmp              2021.3.0          
ipython                   7.16.1         
ipython_genutils          0.2.0                   
jedi                      0.18.0          
joblib                    1.0.1             
jpeg                      9b                  
keras                     2.6.0                   
keras-preprocessing       1.1.2             
lame                      3.100             
lcms2                     2.12                
libauc                    1.1.6               
libblas                   3.9.0            
libcblas                  3.9.0           
libedit                   3.1.20210714        
libffi                    3.2.1             
libgcc-ng                 9.3.0              
libgfortran-ng            11.2.0               
libgfortran5              11.2.0            
libgomp                   9.3.0            
libiconv                  1.16        
liblapack                 3.9.0       
libpng                    1.6.37     
libstdcxx-ng              9.3.0           
libtiff                   4.1.0             
libuv                     1.42.0      
libwebp                   1.2.0   
lz4-c                     1.9.3           
markdown                  3.3.4    
matplotlib                3.1.3                   
mkl                       2021.3.0   
ncurses                   6.2          
nettle                    3.6                  
networkx                  2.5.1                    
ninja                     1.10.2              
numpy                     1.19.5           
nvidia-ml-py3             7.352.0              
oauthlib                  3.1.1                    
olefile                   0.46               
opencv-python             4.5.3.56        
openh264                  2.1.1             
openssl                   1.0.2u               
opt-einsum                3.3.0              
pandas                    1.1.5                   
parso                     0.8.2                 
pexpect                   4.8.0                
pickleshare               0.7.5                 
pillow                    8.3.1          
pip                       21.2.2     
prompt-toolkit            3.0.20                  
protobuf                  3.18.0                  
psutil                    5.8.0                   
ptyprocess                0.7.0             
pyasn1                    0.4.8                 
pyasn1-modules            0.2.8                  
pygments                  2.10.0                   
pyparsing                 2.4.7                   
python                    3.6.4                
python-dateutil           2.8.2                   
python_abi                3.6                    
pytorch                   1.8.0           
pytorch-metric-learning   0.9.99         
pytz                      2021.1                  
pywavelets                1.1.1                  
readline                  7.0                
requests                  2.26.0                   
requests-oauthlib         1.3.0                    
rsa                       4.7.2                 
scikit-image              0.17.2               
scikit-learn              0.24.2        
scipy                     1.5.4                
seaborn                   0.11.2                 
setuptools                58.0.4        
six                       1.15.0                    
sqlite                    3.33.0              
tensorboard               2.6.0                 
tensorboard-data-server   0.6.1                  
tensorboard-logger        0.1.0                 
tensorboard-plugin-wit    1.8.0                    
tensorflow                2.6.0                
tensorflow-estimator      2.6.0                    
termcolor                 1.1.0                   
threadpoolctl             2.2.0              
tifffile                  2020.9.3                 
timm                      0.4.9                    
tk                        8.6.10              
torchaudio                0.8.0                      
torchvision               0.9.0               
tqdm                      4.62.3           
traitlets                 4.3.3                   
typing_extensions         3.7.4.3                    
urllib3                   1.26.7                   
wcwidth                   0.2.5                    
werkzeug                  2.0.2                  
wheel                     0.37.0             
wrapt                     1.12.1                   
xz                        5.2.5               
zipp                      3.6.0                   
zlib                      1.2.11               
zstd                      1.4.9                
```
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
  --ckpt-path 'checkpoints/COVIDNet-SSL.ckpt'
  --data-dir '../archive'
  --exp-dir './train_folder'
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
  --ckpt-path 'checkpoints/COVIDNet-SSL.ckpt'
  --data-dir '../archive'
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
  --ckpt-path 'checkpoints/COVIDNet-SSL.ckpt'
  --img-path '../archive/test/MIDRC-RICORD-1C-419639-002463-12463-0.png'
  --exp-dir './inference_folder'
```
4. For more options and information:
```
python inference.py --help
```

