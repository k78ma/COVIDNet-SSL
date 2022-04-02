
import os
import math
import utils
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.models as models

from libauc.optimizers import PESG
from libauc.losses import AUCMLoss

from PIL import Image
import pandas as pd
import numpy as np

from torchvision import transforms
import torchvision.transforms.functional as TF

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler



#################################### RESNET ####################################

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9  # == pytorch's default value as well


class BatchNormRelu(nn.Sequential):
    def __init__(self, num_channels, relu=True):
        super().__init__(nn.BatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON), nn.ReLU() if relu else nn.Identity())


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2, bias=bias)

class SelectiveKernel(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(conv(in_channels, 2 * out_channels, stride=stride),
                                       BatchNormRelu(2 * out_channels))
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(conv(out_channels, mid_dim, kernel_size=1), BatchNormRelu(mid_dim),
                                         conv(mid_dim, 2 * out_channels, kernel_size=1))

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)  # 2, B, C, H, W
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)  # 2, B, C, 1, 1
        return (x * F.softmax(m, dim=0)).sum(dim=0)

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)),
                                          # kernel_size = 2 => padding = 1
                                          nn.AvgPool2d(kernel_size=2, stride=stride, padding=0),
                                          conv(in_channels, out_channels, kernel_size=1))
        else:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = BatchNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [conv(in_channels, out_channels, kernel_size=1), BatchNormRelu(out_channels)]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(BatchNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(BatchNormRelu(out_channels * 4, relu=False))
        self.net = nn.Sequential(*ops)

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))

class Blocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList([BottleneckBlock(in_channels, out_channels, stride, sk_ratio, True)])
        self.channels_out = out_channels * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(BottleneckBlock(self.channels_out, out_channels, 1, sk_ratio))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class Stem(nn.Sequential):
    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(BatchNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)

class ResNet(nn.Module):
    def __init__(self, layers, width_multiplier, sk_ratio):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)
        self.fc = nn.Linear(channels_in, 1000)

    def forward(self, x, apply_fc=False):
        h = self.net(x).mean(dim=[2, 3])
        if apply_fc:
            h = self.fc(h)
        return h

class ContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x

def get_resnet(depth=50, width_multiplier=1, sk_ratio=0):  # sk_ratio=0.0625 is recommended
    layers = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}[depth]
    resnet = ResNet(layers, width_multiplier, sk_ratio)
    return resnet, ContrastiveHead(resnet.channels_out)

#################################### MODEL ####################################

class Base_Module:
    def __init__(self, args, verbose):
        
        self.args = args
        self.optim = args.optim
        self.dist = args.dist
        self.device = args.device

        self.n_output = 1
        self.lr_intervals = [50, 75] if (args.n_epochs == 100) else [args.n_epochs // 2]
        self.verbose = verbose
        self.model = None

    def construct_optimizer(self, imratio, train_iters_per_epoch=None):

        # Construct Optimizer & Criterion
        if (self.optim == 'SGD'):
            self.criterion = nn.BCEWithLogitsLoss() if (self.n_output == 1) else nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
            )
          
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.n_epochs)

            '''
            if ((train_iters_per_epoch == 0) and (self.verbose)):
                print('[Warning] Iterations per epoch is zero')

            warmup_steps = train_iters_per_epoch * self.args.warmup_epochs
            total_steps = train_iters_per_epoch * self.args.n_epochs

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True)
            )
            '''
            
        elif (self.optim == 'AUC'):

            if ((imratio == 0) and (self.verbose)):
                print('[Warning] Imratio is zero')

            self.criterion = AUCMLoss(imratio=imratio, device=self.device)
            self.optimizer = PESG(
                self.model,
                a=self.criterion.a,
                b=self.criterion.b,
                alpha=self.criterion.alpha,
                imratio=imratio,
                lr=self.args.lr,
                gamma=self.args.gamma,
                margin=self.args.margin,
                weight_decay=self.args.weight_decay,
                device=self.device
            )

        else:
            raise NotImplementedError

        if (self.verbose):
            print('\nUsing %s Optimizer With %.4f Learning Rate | Imratio : %.4f'%(self.optim, self.args.lr, imratio))

    def get_lr(self):
        if (self.optim == 'SGD'):
            return self.scheduler.get_last_lr()[-1]
        elif (self.optim == 'AUC'):
            return self.optimizer.lr

    # Train
    def train(self, dataloader, epoch):

        losses = utils.AverageMeter()

        # Monitor Uniform Sampler
        class_prob_0 = utils.AverageMeter()
        class_prob_1 = utils.AverageMeter()

        # Switch To Train Mode
        self.model.train()

        _len = len(dataloader) * dataloader.batch_size if (self.dist) else len(dataloader.dataset) 

        if (self.verbose):
            pbar = tqdm(desc='Train Loop', total=_len , dynamic_ncols=True)
    
        # Update Learning Rate (AUC) 
        if ((self.optim == 'AUC') and (epoch in self.lr_intervals)):
            self.optimizer.update_regularizer(decay_factor=10)

        for input, target in dataloader:

            # Add To Monitor
            class_prob_0.update(torch.sum(target == 0) / target.size(0))
            class_prob_1.update(torch.sum(target == 1) / target.size(0))

            input = input.cuda(self.device)
            target = target.cuda(self.device) 

            # Compute Output
            output = self.forward(input)
            if (self.optim == 'AUC'):
                output = torch.sigmoid(output)
            output = output.reshape(target.shape)
            target = target.type(torch.float)

            # Compute Loss
            loss = self.criterion(output, target)

            # Record Loss (Updates Average Meter With The Accuracy Value)
            losses.update(loss.item(), input.size(0))

            # Compute Gradient And Do Optimizer Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.verbose):
                pbar.update(input.size(0))

        # Update Learning Rate (SGD) 
        if (self.optim != 'AUC'):
          self.scheduler.step()

        if (self.verbose):
            pbar.close()    
            print('\nEpoch [ %d / %d ] | Class Probabilities [ %.2f | %.2f ]'%(
                epoch, self.args.n_epochs, class_prob_0.avg, class_prob_1.avg))

        return losses.avg

    # Validate 
    def validate(self, dataloader):

        losses = utils.AverageMeter()

        outputs = []
        targets = []

        # Switch To Evaluation Mode
        self.model.eval()

        if (self.verbose):
            pbar = tqdm(desc='Valid Loop', total=len(dataloader.dataset), dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(dataloader):
                
                input = input.cuda(self.device)
                target = target.cuda(self.device)

                # Compute Output
                output = self.forward(input)
                if (self.optim == 'AUC'):
                    output = torch.sigmoid(output)
                output = output.reshape(target.shape)
                target = target.type(torch.float)
                
                # Compute Loss
                loss = self.criterion(output, target)

                # Record Loss (Updates Average Meter With The Accuracy Value)
                losses.update(loss.item(), input.size(0))

                if (self.optim != 'AUC'):
                    output = torch.sigmoid(output)

                # Add To Total Outputs & Targets
                outputs.append(output.detach().cpu())
                targets.append(target.detach().cpu())

                if (self.verbose):
                    pbar.update(input.size(0))

            if (self.verbose):
                pbar.close()
    
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        probs = outputs
        threshold = utils.get_threshold(targets, probs)
        
        _, _, AUC = utils.get_ROC_AUC(targets, probs)
        preds = (probs >= threshold).astype(int).reshape(-1)
        acc1, precision_scores, recall_scores, f1_scores = utils.get_metrics(targets, preds)

        if (self.verbose):
            print('[ --- Validation --- ]')
            print('Loss      : %.4f'%(losses.avg))
            print('Top 1 Acc : %.4f'%(acc1))
            print('ROC-AUC   : %.4f'%(AUC))
            print ('---------------------') 
            print('Precision [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(precision_scores[0], precision_scores[1], np.mean(precision_scores)))
            print('Recall    [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(recall_scores[0], recall_scores[1], np.mean(recall_scores)))
            print('F1        [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(f1_scores[0], f1_scores[1], np.mean(f1_scores)))
            
        return losses.avg, acc1, AUC, np.mean(f1_scores), threshold

    # Evaludate
    def evaluate(self, dataloader, model_pth, save_dir):

        # Load Model
        checkpoint = torch.load(model_pth)
        self.model.load_state_dict(checkpoint['state_dict'])
        threshold = checkpoint['threshold']

        outputs = []
        targets = []

        # switch to evaluate mode
        self.model.eval()

        if (self.verbose):
            pbar = tqdm(desc='Test Loop', total=len(dataloader.dataset), dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(dataloader):
                
                input = input.cuda(self.device)
                target = target.cuda(self.device)

                # Compute Output
                output = self.forward(input)
                output = torch.sigmoid(output)

                # Add To Total Outputs & Targets
                outputs.append(output.detach().cpu())
                targets.append(target.detach().cpu())

                if (self.verbose):
                    pbar.update(input.size(0))
            
            if (self.verbose):
                pbar.close()
    
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        if (self.n_output == 1):
            probs = outputs
        else:
            probs = outputs[:, 1]

        preds = (probs >= threshold).astype(int).reshape(-1)

        title = 'Confusion Matrix | Threshold [%.4f]'%(threshold)
        save_confmat(targets, preds, title, save_dir)

        acc1, precision_scores, recall_scores, f1_scores = save_metrics(targets, preds, save_dir)

        AUC = save_ROC_curve(targets, probs, save_dir)

        labels = {
            'xlabel' : 'Output Probability',
            'ylabel' : 'Density',
            'title' : 'Probability Histogram',
        }
        save_histogram(probs, labels, os.path.join(save_dir, 'histogram.png'))

        if (self.verbose):
            print('Model Name : %s'%(model_pth))
        
        return acc1, precision_scores, recall_scores, f1_scores, AUC

class Resnet50_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading SimCLR-Resnet50 CXR Model...')

        self.model, _ = get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, (self.model.fc.in_features // 2)),
            nn.ReLU(),
            nn.Linear((self.model.fc.in_features // 2), self.n_output)
        )
        assert(os.path.exists(args.ckpt_path))

        state_dict = torch.load(args.ckpt_path)['state_dict']
        msg = self.model.load_state_dict(state_dict, strict=False)
        #assert(msg.missing_keys == ['fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias'])
        if (verbose):
            print('Loaded SimCLR-Resnet50 CXR Model From Checkpoint: %s'%(args.ckpt_path))
        
        self.model = self.model.cuda(self.device)

        self.construct_optimizer(imratio)

    def forward(self, x):
        out = self.model(x, apply_fc=True)
        return out

################################################## DATA HANDLER #####################################################

class_map = {
    'negative' : 0,
    'positive' : 1,
}

def _process_txt_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
        files = [x.split()[-3:-1] for x in files]

    return files

def _process_csv_file(file):
    data = pd.read_csv(file)
    data = data.loc[data['Frontal/Lateral'] == 'Frontal'] # Keep only frontal x-rays
    data = data['Path']

    return list(data)

####### Transformations ######

# Normalization Used On The Image, ImageNet Normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return TF.pad(image, padding, 0, 'constant')

class Transform_Finetuning:
    def __init__(self, img_size=224):

        self.transform_train = transforms.Compose([
            #transforms.RandomRotation(degrees=(-20, 20)),
            SquarePad(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), # Cropped Because Sometimes Resize Doesnt Perfectly Produce 224x224
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
         ])

        self.transform_valid = transforms.Compose([
            SquarePad(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), # Cropped Because Sometimes Resize Doesnt Perfectly Produce 224x224
            transforms.ToTensor(),
            normalize,
        ])

        self.transform_cam = transforms.Compose([
            SquarePad(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), # Cropped Because Sometimes Resize Doesnt Perfectly Produce 224x224
            transforms.ToTensor(),
        ])

    def __call__(self, img, mode='train'):
        
        if (mode == 'train'):
            return self.transform_train(img)
        elif (mode == 'valid'):
            return self.transform_valid(img)
        elif (mode == 'cam'):
            return self.transform_cam(img)


###### Finetuning Data Handlers ######

class Dataset_CovidX_Finetuning(Dataset):
    def __init__(self, img_paths, img_labels, is_train=True):
        super().__init__()

        self.img_paths = img_paths
        self.img_labels = img_labels
        self.is_train = is_train

        # Calculate Imbalance Ratio (For AUC)
        self.imratio = {L : len(np.where(self.img_labels == L)[0]) / self.img_labels.shape[0] for L in np.unique(self.img_labels)}
        
        # Calculate Weights For Uniform Sampling
        n_class_samples = np.array([len(np.where(self.img_labels == L)[0]) for L in np.unique(self.img_labels)])
        class_weights = 1.0 / n_class_samples
        self.sample_weights = np.array([class_weights[L] for L in self.img_labels])
        self.sample_weights = torch.from_numpy(self.sample_weights).float()

        self.transform = Transform_Finetuning()

    def __len__(self):
        return self.img_paths.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if (self.is_train):
            img = self.transform(image, mode='train')
        else:
            img = self.transform(image, mode='valid')

        label = int(self.img_labels[idx])
        
        return img, label

class DataHandler_CovidX_Finetuning(): 
    def __init__(self, data_dir, val_split=0.1, verbose=True):
        
        self.data_dir = data_dir
        self.val_split = val_split 
        self.verbose = verbose
        self.im_count = {}

        if (verbose):
            print('\nClassifying Negative/Positive')

        # Read CovidX Train And Test Files
        train_files = _process_txt_file(os.path.join(self.data_dir, "train_8B.txt"))
        test_files = _process_txt_file(os.path.join(self.data_dir, "test_8B.txt"))

        train_files = [[os.path.join(self.data_dir, 'train', pth), label] for pth, label in train_files]
        test_files = [[os.path.join(self.data_dir, 'test', pth), label] for pth, label in test_files]

        train_files = np.asarray(train_files)
        test_files = np.asarray(test_files)

        # create validation split
        val_files = None
        if self.val_split > 0.0:
            order = np.random.permutation(train_files.shape[0])
            cut_off = int(train_files.shape[0] * (1.0 - self.val_split))
            
            val_files = train_files[order[cut_off:]]
            train_files = train_files[order[:cut_off]]

        # Seperate Data Into Train/Test/Valid
        self.train_img_paths, self.train_labels = self.seperate_data(
            task='train',
            files=train_files
        )

        self.test_img_paths, self.test_labels = self.seperate_data(
            task='test',
            files=test_files
        )

        self.val_img_paths = None
        self.val_labels = None
        if (val_files is not None):
            self.val_img_paths, self.val_labels = self.seperate_data(
                task='valid',
                files=val_files
            )

    def seperate_data(self, task, files):
        
        self.im_count[task] = {
            'negative' : 0,
            'positive' : 0
        }

        img_paths = []
        labels = []

        for fname, label in files:
            img_paths.append(fname)

            self.im_count[task][label] += 1
            labels.append(class_map[label])

        if (self.verbose):
            print("\nnumber of negative cases in %s split: "%(task), self.im_count[task]['negative'])
            print("number of positive cases in %s split: "%(task), self.im_count[task]['positive'])
            
        return np.asarray(img_paths), np.asarray(labels)

    def get_datasets(self):

        train_dataset = Dataset_CovidX_Finetuning(self.train_img_paths, self.train_labels, is_train=True)
        test_dataset = Dataset_CovidX_Finetuning(self.test_img_paths, self.test_labels, is_train=False)
        val_dataset = None
        if ((self.val_img_paths is not None) and (self.val_labels is not None)):
            val_dataset = Dataset_CovidX_Finetuning(self.val_img_paths, self.val_labels, is_train=False)

        return train_dataset, test_dataset, val_dataset
    
    def get_dataloaders(self, args):
    
        train_dataset, test_dataset, val_dataset = self.get_datasets()
        
        sampler = None
        if (args.uniform):
            
            # Uniform Sampling
            sampler = WeightedRandomSampler(
                train_dataset.sample_weights, 
                len(train_dataset.sample_weights), 
                replacement=True
            )

            if (self.verbose):
                print('\nEnabling Uniform Sampling')
        else:

            if (self.verbose):
                print('\nDisabling Uniform Sampling')

        if (args.dist):
            sampler = DistributedSampler(train_dataset) if (sampler is None) else DistributedSamplerWrapper(sampler)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
        )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            sampler=None,
            shuffle=False
        )
        
        val_dataloader = None
        if (val_dataset is not None):
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                sampler=None,
                shuffle=False
            )
        
        return train_dataloader, test_dataloader, val_dataloader, sampler

class Dataset_CovidX_GradCAM(Dataset):
    def __init__(self, img_paths, img_labels):
        super().__init__()

        self.img_paths = img_paths
        self.img_labels = img_labels

        # Calculate Imbalance Ratio (For AUC)
        self.imratio = {L : len(np.where(self.img_labels == L)[0]) / self.img_labels.shape[0] for L in np.unique(self.img_labels)}
        
        # Calculate Weights For Uniform Sampling
        n_class_samples = np.array([len(np.where(self.img_labels == L)[0]) for L in np.unique(self.img_labels)])
        class_weights = 1.0 / n_class_samples
        self.sample_weights = np.array([class_weights[L] for L in self.img_labels])
        self.sample_weights = torch.from_numpy(self.sample_weights).float()

        self.transform = Transform_Finetuning()

    def __len__(self):
        return self.img_paths.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.split(img_path)[-1]
        
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image, mode='cam')
        input = self.transform(image, mode='valid')

        label = int(self.img_labels[idx])
        
        return img_name, img, input, label

class DataHandler_CovidX_GradCAM():
    def __init__(self, data_dir, val_split=0.1, verbose=True):
        
        self.data_dir = data_dir
        self.val_split = val_split 
        self.verbose = verbose
        self.im_count = {}


        if (verbose):
            print('\nClassifying Negative/Positive')

        # Read CovidX Train And Test Files
        train_files = _process_txt_file(os.path.join(self.data_dir, "train_8B.txt"))
        test_files = _process_txt_file(os.path.join(self.data_dir, "test_8B.txt"))

        train_files = [[os.path.join(self.data_dir, 'train', pth), label] for pth, label in train_files]
        test_files = [[os.path.join(self.data_dir, 'test', pth), label] for pth, label in test_files]

        train_files = np.asarray(train_files)
        test_files = np.asarray(test_files)

        # create validation split
        val_files = None
        if self.val_split > 0.0:
            order = np.random.permutation(train_files.shape[0])
            cut_off = int(train_files.shape[0] * (1.0 - self.val_split))
            
            val_files = train_files[order[cut_off:]]
            train_files = train_files[order[:cut_off]]

        # Seperate Data Into Train/Test/Valid
        self.train_img_paths, self.train_labels = self.seperate_data(
            task='train',
            files=train_files
        )

        self.test_img_paths, self.test_labels = self.seperate_data(
            task='test',
            files=test_files
        )

        self.val_img_paths = None
        self.val_labels = None
        if (val_files is not None):
            self.val_img_paths, self.val_labels = self.seperate_data(
                task='valid',
                files=val_files
            )

    def seperate_data(self, task, files):
        
        self.im_count[task] = {
            'negative' : 0,
            'positive' : 0
        }

        img_paths = []
        labels = []

        for fname, label in files:
            img_paths.append(fname)

            self.im_count[task][label] += 1
            labels.append(class_map[label])

        if (self.verbose):
            print("\nnumber of negative cases in %s split: "%(task), self.im_count[task]['negative'])
            print("number of positive cases in %s split: "%(task), self.im_count[task]['positive'])
            
        return np.asarray(img_paths), np.asarray(labels)

    def get_datasets(self):

        train_dataset = Dataset_CovidX_GradCAM(self.train_img_paths, self.train_labels)
        test_dataset = Dataset_CovidX_GradCAM(self.test_img_paths, self.test_labels)
        val_dataset = None
        if ((self.val_img_paths is not None) and (self.val_labels is not None)):
            val_dataset = Dataset_CovidX_GradCAM(self.val_img_paths, self.val_labels)

        return train_dataset, test_dataset, val_dataset

    def get_dataloaders(self):
    
        train_dataset, test_dataset, val_dataset = self.get_datasets()
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=128, 
            sampler=None,
            shuffle=False
        )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=1, 
            sampler=None,
            shuffle=False
        )
        
        val_dataloader = None
        if (val_dataset is not None):
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=128, 
                sampler=None,
                shuffle=False
            )
        
        return train_dataloader, test_dataloader, val_dataloader