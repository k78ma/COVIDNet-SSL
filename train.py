
import os
import time
import shutil

import utils
import argparse
from tqdm import trange

import torch

from model_loader import Resnet50_Module

from model_loader import DataHandler_CovidX_Finetuning

import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='COVID-Net SimCLR-CXR Training Script')

parser.add_argument('--n-epochs', default=1, type=int, help='Number of epochs')
parser.add_argument('--ckpt-path', default='checkpoints/COVIDNet-SSL.ckpt', 
    type=str, help='path to pretrained model checkpoint')
parser.add_argument('--data-dir', default="../archive", 
    type=str, help='path to dataset')
parser.add_argument('--exp-dir', default="./train_folder", 
    type=str, help='export directory')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--batch-size', default=128, 
    type=int, help='batch size')
parser.add_argument('--optim', default='SGD',
    type=str, choices=['SGD', 'AUC'], help='which optimizer to use')
parser.add_argument('--pretrained', default='SimCLR',
    type=str, help='pre-training architecture')

parser.add_argument('--seed', default=123, 
    type=int, help='seed for dataset generation')
parser.add_argument('--weight-decay', default=1e-4, 
    type=float, help='weight decay for embedding and classifier parameters') # Only Used For SGD Optimization
parser.add_argument('--gamma', default=500,
        type=int, help='gamma for AUC maximization') # Only Used For AUC Maximization
parser.add_argument('--margin', default=1.0,
    type=float, help='margin for AUC maximization') # Only Used For AUC Maximization
parser.add_argument('--uniform', 
    action='store_true', help='enable uniform sampling')

parser.add_argument('--dist', 
    action='store_true', help='enable for distributed training')
parser.add_argument('--num-nodes', default=1, 
    type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, 
    type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1234', 
    type=str, help='url used for distributed training')


def setup(ngpus_per_node, args):
    args.rank = args.rank * ngpus_per_node + args.device
    dist.init_process_group('nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

def cleanup():
    dist.destroy_process_group()

def main(gpu, ngpus_per_node, args):

    # Distributed Training Setup 
    args.device = gpu
    torch.cuda.device(args.device)
    
    if (args.dist):
        setup(ngpus_per_node, args)
    else:
        assert((ngpus_per_node == 1) and (args.rank == 0))
    
    # Only Output On One Device
    localmaster = (args.rank == 0)

    #################################### Prepare Logger & Output Directory ##################################################

    if (localmaster):
        utils.prepare_directory(args.exp_dir)

        logger_titles = [
            'Best ROC-AUC',
            'Best F1',
            'Learning Rate',
            'Training Loss',
            'Validation Loss',
            'Validation ROC-AUC',
            'Validation F1',
            'Threshold',
        ]

        log_dir = os.path.join(args.exp_dir, "training_logs")
        utils.prepare_directory(log_dir)
        logger = utils.Logger(log_dir, 'logs', logger_titles)

    ############################################# Setup DataLoaders #########################################################

    args.batch_size = int(args.batch_size / ngpus_per_node)
    train_dataloader, test_dataloader, val_dataloader, sampler = DataHandler_CovidX_Finetuning(
        args.data_dir, 
        verbose=localmaster
    ).get_dataloaders(args)

    #################################### Load Model, Optimizer, Scheduler & Criterion #######################################
    
    # Load Model
    imratio = train_dataloader.dataset.imratio[1]
    #train_iters_per_epoch = len(train_dataloader.dataset) // (args.batch_size * ngpus_per_node * args.num_nodes)
    
    model = Resnet50_Module(
        args, 
        verbose=True,
        imratio=imratio,
    )

    #################################### Begin Finetuning ##################################################################

    start_epoch = 0
    if (localmaster):
        if (args.dist):
            print("\nStarting %s Distributed Finetuning From Epoch %d ..."%(args.pretrained, start_epoch))
        else:
            print("\nStarting %s Finetuning From Epoch %d ..."%(args.pretrained, start_epoch))
        print("-" * 100)

    lowest_loss = 100
    best_acc1 = 0
    best_AUC = 0
    best_f1 = 0
    for epoch in trange(start_epoch, args.n_epochs, desc='epoch_monitor', dynamic_ncols=True):
        
        print('\n')

        # Necessary For Random Batch Resampling
        try:
            sampler.set_epoch(epoch)
        except:
            pass

        train_loss = model.train(train_dataloader, epoch)
        lr = model.get_lr()

        if (localmaster):
            val_loss, acc1, AUC, average_f1, threshold = model.validate(val_dataloader)

            # Remember State Dict
            state = {
                'state_dict': model.module.model.state_dict() if (args.dist) else model.model.state_dict(),
                'optimizer': model.optimizer.state_dict(),
                'scheduler': model.scheduler.state_dict() if hasattr(model, 'scheduler') else None,
                'epoch': epoch,
                'threshold': threshold,
            }

            # Save Models
            default_path = os.path.join(args.exp_dir, 'model_last_epoch.ckpt')
            torch.save(state, default_path)

            if (acc1 > best_acc1):
                shutil.copyfile(default_path, os.path.join(args.exp_dir, 'model_best_acc.ckpt'))
                best_acc1 = acc1

            if (AUC > best_AUC):
                shutil.copyfile(default_path, os.path.join(args.exp_dir, 'model_best_auc.ckpt'))
                best_AUC = AUC

            if (average_f1 > best_f1):
                shutil.copyfile(default_path, os.path.join(args.exp_dir, 'model_best_f1.ckpt'))
                best_f1 = average_f1

            if (val_loss < lowest_loss):
                shutil.copyfile(default_path, os.path.join(args.exp_dir, 'model_lowest_loss.ckpt'))
                lowest_loss = val_loss

            # Append logger file
            logger.append([
                best_AUC,
                best_f1,
                lr,
                train_loss,
                val_loss,
                AUC,
                average_f1,
                threshold,
            ], step=epoch)

            print('Training Loss: %.3f | Validation Loss : %.3f | ROC-AUC : %.3f'%(train_loss, val_loss, AUC))

    if (localmaster):
        logger.close()
        print("Finetuning Complete. Final Loss: {:.2f}".format(train_loss))

if __name__ == "__main__":

    args = parser.parse_args()

    utils.set_seed(args.seed)

    start_time = time.time() 
    if (args.dist):

        # Find GPUS & Setup Parameters
        ngpus_per_node = torch.cuda.device_count()
        assert (ngpus_per_node >= 2), 'Requires at least 2 GPUs, but found only %d'%(ngpus_per_node)
        args.world_size = ngpus_per_node * args.num_nodes

        mp.spawn(
            main, 
            args=(ngpus_per_node, args), 
            nprocs=ngpus_per_node
        )
    else:
        
        # Find Free GPU
        free_gpu = utils.get_free_gpu()

        main(free_gpu, 1, args)
    
    end_time = time.time() 
    print("\nTotal Time Elapsed: {:.2f}h".format((end_time - start_time) / 3600.0))

