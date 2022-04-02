
import os
import time
import argparse
import utils

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

import model_loader
from model_loader import get_resnet, DataHandler_CovidX_Finetuning, SquarePad

parser = argparse.ArgumentParser(description='COVID-Net SimCLR-CXR Inference Script')

parser.add_argument('--ckpt-path', default='checkpoints/SSL976.ckpt', 
    type=str, help='path to model checkpoint')
parser.add_argument('--img-path', default='../archive/test/MIDRC-RICORD-1C-419639-002463-12463-0.png',
    type=str, help='path to test image')
parser.add_argument('--exp-dir', default="./sample_inf_folder", 
    type=str, help='export directory')
parser.add_argument('--seed', default=123, 
    type=int, help='seed for dataset generation')


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform_test = transforms.Compose([
    model_loader.SquarePad(),
    transforms.Resize(224),
    transforms.CenterCrop(224), # Cropped Because Sometimes Resize Doesnt Perfectly Produce 224x224
    transforms.ToTensor(),
    normalize,
])


def main(gpu, img_path, exp_dir):

    utils.prepare_directory(exp_dir)

    # Distributed Training Setup 
    device = gpu
    
    # Only Output On One Device
    localmaster = True

    ############################################# Setup DataLoaders #########################################################
  
    image = Image.open(img_path).convert('RGB')
    image = transform_test(image)
    image = image.unsqueeze(0)
    image = image.cuda(device)

    #################################### Load Model, Optimizer, Scheduler & Criterion #######################################


    # Load Model
    model, _= get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
        
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, (model.fc.in_features // 2)),
        nn.ReLU(),
        nn.Linear((model.fc.in_features // 2), 1)
    )
    
    model_pth = args.ckpt_path
    checkpoint = torch.load(model_pth, map_location='cuda:%d'%(device))
    model.load_state_dict(checkpoint['state_dict'])
    threshold = checkpoint['threshold']
    #threshold = 0.5

    model = model.cuda()
    

#################################### Begin Testing ##################################################################
    
    # Switch To Evaluation Mode
    model.eval()

    output = model(image, apply_fc = True)
    output = torch.sigmoid(output)
    output = output.reshape(torch.Size([1]))

    if (output < threshold):
        prediction = "0 (Negative)"
    else:
        prediction = "1 (Positive)"
        
    with open(os.path.join(exp_dir, 'inference.txt'), 'w') as file:
        file.write('Model: {}'.format(args.ckpt_path))
        file.write('\n')
        file.write('\n[ --- Inference --- ]')
        file.write('\nPrediction: {}'.format(prediction))
        file.write('\n')
        file.write('\n**DISCLAIMER**')
        file.write('\nDo not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')

if __name__ == "__main__":

    args = parser.parse_args()

    utils.set_seed(args.seed)

    start_time = time.time() 

    # Find Free GPU
    free_gpu = utils.get_free_gpu()
    main(free_gpu, args.img_path, args.exp_dir)

    end_time = time.time() 
    print("\nTotal Time Elapsed: {:.2f}h".format((end_time - start_time) / 3600.0))