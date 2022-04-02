
import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import os
import json
import shutil
import random
import datetime
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorboard_logger import Logger as TBLogger

import torch

from libauc.optimizers import PESG
from libauc.losses import AUCMLoss 

import seaborn as sn
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KernelDensity

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.__sum = 0
        self.__count = 0

    def update(self, val, n=1):
        self.val = val
        self.__sum += val * n
        self.__count += n

    @property
    def avg(self):
        if self.__count == 0:
            return 0.
        return self.__sum / self.__count

class Logger(object):

    def __init__(self, log_dir, label, titles, append_steps=1):
        """
        log_dir      : str, directory where all the logs will be written.
        label        : str, root filename for the logs. It shouldn't contain an extension, such as .txt
        titles       : list, title for each log attribute.
        append_steps : int, 
        """

        self.log_dir = log_dir
        self.label = label
        self.titles = titles
        self.append_steps = append_steps

        self.logs = {} # all title-log pairs that will be traced for this instance
        self.meters = {}
        for t in titles:
            self.logs[t] = []
            self.meters[t] = AverageMeter()

        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        self.tb_logger = TBLogger(self.log_dir)
        self.f_txt = open(os.path.join(self.log_dir, '{}.txt'.format(self.label)), 'w')

    def flush(self):
        self.save_as_arrays()
        self.save_as_figures()

    def close(self):
        self.flush()
        self.f_txt.close()

    def update(self, values, step):
        """
        Adds a new log value for each title, also updates corresponding average meters.
        If step is multiple of append_steps, then self.append is called.

        values : list, must be of the same size as self.titles.
        step   : int, a step number
        """
        assert len(self.titles) == len(values)

        for t, v in zip(self.titles, values):
            self.meters[t].update(v, 1)

        if step % self.append_steps == 0:
            values = [m.avg for m in self.meters.values()]
            self.append(values, step)

    def append(self, values, step):
        """
        Adds a new log value for each title.

        values : list, must be of the same size as self.titles.
        step   : int, a step number
        """
        assert len(self.titles) == len(values)
        
        step_log = OrderedDict()
        step_log['step'] = str(step)
        step_log['time'] = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

        for t, v in zip(self.titles, values):
            self.logs[t].append(v)
            step_log[t] = v
            self.tb_logger.log_value(t, v, step)

        json.dump(step_log, self.f_txt, indent=4)
        self.f_txt.write('\n')
        self.f_txt.flush()

    def save_as_arrays(self):
        """
        Converts all logs to numpy arrays and saves them into self.log_dir.
        """
        arrays = {}
        for t, v in self.logs.items():
            v = np.array(v)
            arrays[t] = v

        np.savez(os.path.join(self.log_dir, '{}.npz'.format(self.label)), **arrays)

    def save_as_figures(self):
        """
        First, converts all logs to numpy arrays, then plots them using matplotlib. Finally, saves the plots into self.log_dir.
        """
        for t, v in self.logs.items():
            v = np.array(v)

            fig = plt.figure(dpi=400)
            ax = fig.add_subplot(111)
            ax.plot(v)
            ax.set_title(t)
            ax.grid(True)
            fig.savefig(
                os.path.join(self.log_dir, '{}_{}.png'.format(self.label, t)),
                bbox_inches='tight' )
            plt.close()

def prepare_directory(directory, force_delete=False):
    if os.path.exists(directory) and not force_delete:
        print ('directory: %s already exists, backing up this folder ... ' % directory)
        backup_dir = directory + '_backup'

        if os.path.exists(backup_dir):
            print ('backup directory also exists, removing the backup directory first')
            shutil.rmtree(backup_dir, True)

        shutil.copytree(directory, backup_dir)

    shutil.rmtree(directory, True)

    os.makedirs(directory)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    

def get_free_gpu():
    
    try:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        free_gpu = np.argmax(memory_available)
        
        print('\nRunning On GPU %d'%(free_gpu))

        return free_gpu
    except:

        print('\nFailed To Execute Command [nvidia-smi], Defaulting To GPU 0')
        return 0

def get_threshold(targets, probs):

    thresholds = sorted(np.unique(probs))
    f1_scores = []
    for t in thresholds:
        bin_preds = (probs >= t).astype(int).reshape(-1)
        f1_scores.append(f1_score(targets, bin_preds, average=None, zero_division=0))

    best_idx = np.argmax(np.mean(f1_scores, axis=1))
    optimal_threshold = thresholds[best_idx]
    assert((optimal_threshold > 0.0) and (optimal_threshold < 1.0))
    
    return np.float(optimal_threshold)

def get_metrics(targets, preds):

    acc1 = sum(targets == preds) / targets.shape[0]
    precision_scores = precision_score(targets, preds, average=None, zero_division=0, pos_label=1)
    recall_scores = recall_score(targets, preds, average=None, zero_division=0, pos_label=1)
    f1_scores = f1_score(targets, preds, average=None, zero_division=0, pos_label=1)

    return acc1, precision_scores, recall_scores, f1_scores

def get_ROC_AUC(targets, probs):
    
    # Compute AUC
    fpr, tpr, _ = metrics.roc_curve(targets, probs, pos_label=1)
    AUC = metrics.roc_auc_score(targets, probs)

    return fpr, tpr, AUC

def save_confmat(targets, preds, title, save_dir):
    confmat = confusion_matrix(targets, preds, labels=np.arange(2))
    df_cm = pd.DataFrame(confmat, range(2), range(2))
    plt.figure()
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt="d")
    plt.title(title)
    plt.savefig(os.path.join(save_dir, 'confmat.png'))
    plt.close()

def save_metrics(targets, preds, save_dir):
    acc1, precision_scores, recall_scores, f1_scores = get_metrics(targets, preds)

    outfile = open(os.path.join(save_dir, 'metrics.txt'), "w")
    file_content = [
        "Negative Precision: {:.3f}, Positive Precision: {:.3f}\n".format(precision_scores[0], precision_scores[1]),
        "Negative Recall: {:.3f}, Positive Recall: {:.3f}\n".format(recall_scores[0], recall_scores[1])
    ]
    outfile.writelines(file_content)
    outfile.close()

    return acc1, precision_scores, recall_scores, f1_scores

def save_ROC_curve(targets, probs, save_dir):
    fpr, tpr, AUC = get_ROC_AUC(targets, probs)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ResNet-50 (Area = %0.3f)'%(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc-auc.png'))
    plt.close()
    np.savez(os.path.join(save_dir, 'roc-data'), fpr=fpr, tpr=tpr)

    return AUC

def save_histogram(probs, labels, save_pth):
    
    probs = np.array(probs).reshape(-1, 1)
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(probs)
    x_d = np.linspace(0, 1, 1000)
    logprob = kde.score_samples(x_d.reshape(-1, 1))

    plt.figure()
    plt.fill_between(x_d, np.exp(logprob), edgecolor='black', alpha=0.5)
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.title(labels['title'])
    plt.savefig(save_pth)
    plt.close()
    
    '''
    plt.figure()
    plt.hist(probs, len(probs), color='navy', range=[0, 1], edgecolor='none')
    plt.xlabel('Output Probability')
    plt.ylabel('Number of Predictions')
    plt.title('Probability Histogram')
    plt.savefig(os.path.join(save_dir, 'histogram.png'))
    plt.close()
    '''
