import os
import time
import shutil

import utils
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model_loader import get_resnet
from model_loader import Resnet50_Module

from model_loader import DataHandler_CovidX_GradCAM


parser = argparse.ArgumentParser(description='COVID-Net SimCLR-CXR Evaluation Script')

parser.add_argument('--ckpt-path', default='checkpoints/SSL976.ckpt', 
    type=str, help='path to model checkpoint')
parser.add_argument('--data-dir', default="../archive", 
    type=str, help='path to dataset')
parser.add_argument('--exp-dir', default="./evalfolder", 
    type=str, help='export directory')
parser.add_argument('--optim', default='SGD',
    type=str, choices=['SGD', 'Adam'], help='which optimizer to use')

parser.add_argument('--seed', default=123, 
    type=int, help='seed for dataset generation')
parser.add_argument('--uniform', 
    action='store_true', help='enable uniform sampling')
parser.add_argument('--dist', 
    action='store_true', help='enable for distributed training')

def main(gpu, data_dir, exp_dir):

    utils.prepare_directory(exp_dir)

    # Distributed Training Setup 
    device = gpu
    
    # Only Output On One Device
    localmaster = True

    ############################################# Setup DataLoaders #########################################################

    train_dataloader, test_dataloader, val_dataloader = DataHandler_CovidX_GradCAM(
        args.data_dir, 
        verbose=localmaster
    ).get_dataloaders()
    
    if (args.optim == 'SGD'):
        criterion = nn.BCEWithLogitsLoss()
    elif(args.optim == 'AUC'):
        criterion = AUCMLoss(imratio=imratio, device=device)
        
    
################################################### Final Evaluation ####################################################

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
    
    losses = utils.AverageMeter()
    
    outputs = []
    targets = []
    
    model.eval()
    
    print('\nEvaluating model in: {ckpt_path}'.format(ckpt_path=args.ckpt_path))
    
    pbar = tqdm(desc='Test Loop', total=len(test_dataloader.dataset), dynamic_ncols=True)

    negatives = []
    n_negatives = []
    positives = []
    n_positives = []
    trust_scores = []

    with torch.no_grad():
        for batch_idx, (_, _, input, target) in enumerate(test_dataloader):
            
            input = input.cuda(device)
            target = target.cuda(device)
    
            # Compute Output
            output = model(input, apply_fc=True)
            output = torch.sigmoid(output)
            output = output.reshape(target.shape)
            #print(target.shape)
            target = target.type(torch.float)
    
            # Add To Total Outputs & Targets
            outputs.append(output.detach().cpu())
            targets.append(target.detach().cpu())
            
            #print("Predicted class is: {}".format(output))
            
            if (output < threshold):
                negatives.append(output.item())
                n_negatives.append((target == 0))
            else:
                positives.append(output.item())
                n_positives.append((target == 1))
    
            pbar.update(input.size(0))
    
        pbar.close()
    
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        probs = outputs
        threshold = utils.get_threshold(targets, probs)
        
        _, _, AUC = utils.get_ROC_AUC(targets, probs)
        preds = (probs >= threshold).astype(int).reshape(-1)
        acc1, precision_scores, recall_scores, f1_scores = utils.get_metrics(targets, preds)
        
        negatives = np.array(negatives)
        n_negatives = np.array(n_negatives)
        positives = np.array(positives)
        n_positives = np.array(n_positives)

        ######## Scale Negatives #########

        old_min = min(negatives)
        old_max = max(negatives)

        new_min = 0.0
        new_max = 0.5
        
        old_range = old_max - old_min
        new_range = new_max - new_min

        negatives = ((negatives - old_min) * new_range / old_range) + new_min

        ####### Scale Positives #########

        old_min = min(positives)
        old_max = max(positives)

        new_min = 0.5
        new_max = 1.0

        old_range = old_max - old_min
        new_range = new_max - new_min

        positives = ((positives - old_min) * new_range / old_range) + new_min

        #################################
        
        confidences = []
        
        for i in range(positives.shape[0]):
            score = positives[i]
            correct = n_positives[i]

            if (correct):
                confidences.append(score) 
            else:
                confidences.append(1 - score)

        for i in range(negatives.shape[0]):
            score = negatives[i]
            correct = n_negatives[i]

            if (correct):
                confidences.append(1 - score) 
            else:
                confidences.append(score)

        trust_scores.append(np.mean(confidences))

        with open(os.path.join(exp_dir, 'evaluation.txt'), 'w') as file:
            file.write('Model: {}'.format(args.ckpt_path))
            file.write('\n')
            file.write('\n[ --- Evaluation --- ]')
            file.write('\nTop 1 Acc : %.4f'%(acc1))
            file.write('\nROC-AUC   : %.4f'%(AUC))
            file.write('\n---------------------') 
            file.write('\nPrecision [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(precision_scores[0], precision_scores[1], np.mean(precision_scores)))
            file.write('\nRecall    [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(recall_scores[0], recall_scores[1], np.mean(recall_scores)))
            file.write('\nF1        [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(f1_scores[0], f1_scores[1], np.mean(f1_scores)))
            file.write('\n')
            file.write('\nTrust Score: %.4f'%(np.mean(trust_scores)))

        title = 'Confusion Matrix | Threshold [%.4f]'%(threshold)
        utils.save_confmat(targets, preds, title, exp_dir)

        AUC = utils.save_ROC_curve(targets, probs, exp_dir)

        labels = {
            'xlabel' : 'Output Probability',
            'ylabel' : 'Density',
            'title' : 'Probability Histogram',
        }
        utils.save_histogram(probs, labels, os.path.join(exp_dir, 'histogram.png'))
        
if __name__ == "__main__":

    args = parser.parse_args()

    utils.set_seed(args.seed)

    start_time = time.time() 

    # Find Free GPU
    free_gpu = utils.get_free_gpu()
    main(free_gpu, args.data_dir, args.exp_dir)

    end_time = time.time() 
    print("\nTotal Time Elapsed: {:.2f}h".format((end_time - start_time) / 3600.0))


