#Import Necessary Libraries

import os
import time
import random
import numpy as np
import pandas as pd

import torch

from sklearn.model_selection import KFold

from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from tqdm import tqdm_notebook

from config import Config
from efficientnet import Net
from dataset import DatasetRetriever
from metrics import AverageMeter, RocAucMeter
from loss import criterion_margin_focal_binary_cross_entropy
from augmentation import train_augmentaions, validation_augmentations

# Handle Warnings 

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Set Seed

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

#Load data

df_1=pd.read_csv("./train.csv",index_col="image_name")
df_2=pd.read_csv("./train2.csv",index_col="image_name")
df_3=df_1.append(df_2)
di = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:0,16:1,17:2,18:3,19:4,20:5,21:6,22:7,23:8,24:9,25:10,26:11,27:12,28:13,29:14}
df_3 = df_3.replace({'tfrecord':di})
img_dir="/content/train"

# Engine

class Engine:
    
    def __init__(self,model,device,config,fold,model_name='b0',image_size=384, weight_path='./'):
  
        self.model=model
        self.device=device
        self.config=config
        self.best_score=0
        self.best_loss=5000
        self.fold=fold
        self.model_name = model_name
        self.image_size = image_size
        self.weight_path = weight_path
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.criterion = criterion_margin_focal_binary_cross_entropy().to(self.device)
        
    def fit(self,train_loader,validation_loder):

        for epoch in range(self.config.n_epoches):

            print("Training Started...")

            t=time.time()
            summary_loss, roc_auc_scores = self.train_one_epoch(train_loader)

            print('Train : Epoch {:03}: | Summary Loss: {:.3f} | ROC_AUC: {:.3f} | Training time: {}'.format(epoch,summary_loss.avg,roc_auc_scores.avg,time.time() - t))
            

            print("Validation Started...")

            t=time.time()
            summary_loss, roc_auc_scores = self.validation(validation_loder)

            print('Valid : Epoch {:03}: | Summary Loss: {:.3f} | ROC_AUC: {:.3f} | Training time: {}'.format(epoch,summary_loss.avg,roc_auc_scores.avg,time.time() - t))
            
            self.scheduler.step(metrics=roc_auc_scores.avg)
            
            # Early Stopping

            if not self.best_score:

                self.best_score = roc_auc_scores.avg  
        
                print('Saving model with best val as {}'.format(self.best_score))

                self.model.eval()   
                patience = self.config.patience

                torch.save({'model_state_dict': self.model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  f"{self.weight_path}/{self.image_size}_{self.model_name}_{self.fold}.pt")
                continue  

            if roc_auc_scores.avg >= self.best_score:

                self.best_score = roc_auc_scores.avg

                patience = self.config.patience  # Resetting patience since we have new best validation accuracy

                print('Imporved model with best val as {}'.format(self.best_score))

                torch.save({'model_state_dict': self.model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  f"{self.weight_path}/{self.image_size}_{self.model_name}_{self.fold}.pt")
            else:
                patience -= 1

                print('Patience Reduced')

                if patience == 0:
                    print('Early stopping. Best Val roc_auc: {:.3f}'.format(self.best_score))
                    break
                    
    def validation(self, val_loader):

        self.model.eval()
        
        summary_loss = AverageMeter()
        roc_auc_scores = RocAucMeter()

        for _ ,(images, targets) in enumerate(tqdm_notebook(val_loader)):
            with torch.no_grad():

                batch_size = images.shape[0]               
                images = images.to(self.device, dtype=torch.float32)

                targets = targets.to(self.device, dtype=torch.float32)

                outputs = self.model(images)

                loss = self.criterion(outputs, targets)
                
                roc_auc_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, roc_auc_scores


    def train_one_epoch(self, train_loader):

        self.model.train()

        summary_loss = AverageMeter()
        roc_auc_scores = RocAucMeter()

        for _ ,(images, targets) in enumerate(tqdm_notebook(train_loader)):


            batch_size = images.shape[0]               
            images = images.to(self.device, dtype=torch.float32)

            targets = targets.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, targets)
            loss.backward()

            self.optimizer.step()

            roc_auc_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, roc_auc_scores

# Connector

def perform_for_fold(fold_number=0, model_name='b3', image_size=512, weight_path='./', load_weights_path=None):

  skf = KFold(n_splits=5,shuffle=True,random_state=42)

  for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):

      if fold == fold_number:

        print("FOLD : {}".format(fold))

        X_train_id = df_3.loc[df_3.tfrecord.isin(idxT)].index.values
        X_train_label = df_3.loc[df_3.tfrecord.isin(idxT)].target.values

        X_val_id = df_3.loc[(df_3.patient_id!=-1) & (df_3.tfrecord.isin(idxV))].index.values
        X_val_label = df_3.loc[(df_3.patient_id!=-1) & (df_3.tfrecord.isin(idxV))].target.values

        train_gen = DatasetRetriever(
            image_ids=X_train_id,
            img_dir="./",
            labels=X_train_label,
            transforms=train_augmentaions(size=image_size),
        )

        valid_gen = DatasetRetriever(
            image_ids=X_val_id,
            img_dir="./",
            labels=X_val_label,
            transforms=validation_augmentations(size=image_size),
        )

        train_loader = torch.utils.data.DataLoader(
            train_gen,
            sampler=BalanceClassSampler(labels=train_gen.get_labels(), mode="downsampling"),
            batch_size=Config.batch_size,
            pin_memory=False,
            drop_last=True,
            num_workers=Config.num_workers,
        )
        
        validation_loder = torch.utils.data.DataLoader(
            valid_gen, 
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
            shuffle=False,
            sampler=SequentialSampler(valid_gen),
            pin_memory=False,
        ) 
        model = Net(model_name=model_name).cuda()

        if load_weights_path is not None:

          model.load_state_dict(torch.load(load_weights_path + f"{image_size}_{model_name}_{fold_number}.pt")["model_state_dict"]) 
          print("Weight Loaded")

        engine = Engine(model=model, device=torch.device('cuda'), config=Config, fold=fold, model_name=model_name, image_size=image_size, weight_path=weight_path)
        engine.fit(train_loader, validation_loder)
                           
perform_for_fold(fold_number=0, model_name='b6', image_size=512, load_weights_path="./")
perform_for_fold(fold_number=1, model_name='b6', image_size=512, load_weights_path="./")
perform_for_fold(fold_number=2, model_name='b6', image_size=512, load_weights_path="./")
perform_for_fold(fold_number=3, model_name='b6', image_size=512, load_weights_path="./")
perform_for_fold(fold_number=4, model_name='b6', image_size=512, load_weights_path="./")