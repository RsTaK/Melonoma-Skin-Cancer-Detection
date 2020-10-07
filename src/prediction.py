# Import Necessary Libraries

import cv2
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from efficientnet_pytorch import EfficientNet

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler

# Load Data

df_1=pd.read_csv("./train.csv")
df_2=pd.read_csv("./train2.csv")
sample = pd.read_csv('./sample_submission.csv', index_col='image_name')

df_3=df_1.append(df_2)
di = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:0,16:1,17:2,18:3,19:4,20:5,21:6,22:7,23:8,24:9,25:10,26:11,27:12,28:13,29:14}

df_3 = df_3.replace({'tfrecord':di})
df_3 = df_3[df_3.patient_id!=-1]

img_dir="./"

df_test = pd.read_csv('test.csv')

# Data Loader

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, image_ids, labels=None, validation_img_dir=None, test_img_dir=None, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms
        self.validation_img_dir = validation_img_dir
        self.test_img_dir = test_img_dir

    def __getitem__(self, idx: int):
      
      if self.validation_img_dir is not None:

        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.validation_img_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0
        label = self.labels[idx]

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        target = onehot(2, label)
        return image, target
      
      if self.test_img_dir is not None:
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.test_img_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0
        
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)

# Augmentations

def train_augmentaions(image_size=384):
    return A.Compose([
            A.Resize(height=image_size, width=image_size, p=1),
            
            A.RandomSizedCrop(
                min_max_height=(np.floor(image_size*0.8), np.floor(image_size*0.8)), 
                height=image_size, 
                width=image_size, 
                p=0.5), # 20% of height and width to be reduced
            
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(max_holes=8, max_width=64, max_height=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),                  
        ], p=1.0)

def validation_augmentations(image_size=384):
    return A.Compose([
            A.Resize(height=image_size, width=image_size, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

# EfficientNet Model

def Net(model_name):
    model = EfficientNet.from_pretrained(f'efficientnet-{model_name}')
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=2, bias=True)
    return model

def get_model(efficient_net, fold_number, image_size=384, model_weight_path='./'):
    model = Net(efficient_net).cuda()
    model_state_dict = torch.load(f"{model_weight_path}/{image_size}_{efficient_net}_{fold_number}.pt")["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model

# Engine

class Prediction:
    def __init__(self, model, val_img_len, tta=11):
        self.model=model
        self.model.eval()
        
        self.oof_pred = np.zeros((val_img_len, ))
        self.oof_tar = np.zeros((val_img_len, ))
        self.oof_fold = np.zeros((val_img_len, ))
        
        self.test_pred = np.zeros((len(sample),))        
        
        self.tta = tta
        
    def predict(self, validation_loader, test_loader, fold_number):
        with torch.no_grad():
            for i in range(self.tta):
                
                print("Calculating OOF...")
                
                for steps,(images, targets) in enumerate(tqdm(validation_loader)):
                    images = images.cuda()
                    pred = self.model(images) 
                    pred = nn.functional.softmax(pred, dim=1).data.cpu().numpy()[:,1]

                    self.oof_pred[steps*32: (steps+1)*32] += pred
                    self.oof_tar[steps*32: (steps+1)*32] = targets[:,1]
                    self.oof_fold[steps*32: (steps+1)*32] = fold_number
                    
                print("{} TTA performed on OOF prediction..".format(i+1))
                
                print("Predicting for Test Data...")
                
                for steps,images in enumerate(tqdm(test_loader)):
                    images = images.cuda()
                    pred = self.model(images) 
                    pred = nn.functional.softmax(pred, dim=1).data.cpu().numpy()[:,1]
                    
                    self.test_pred[steps*32: (steps+1)*32] += pred
                print("{} TTA performed on Test Data..".format(i+1))   

            self.oof_pred /= self.tta
            self.test_pred /= self.tta
            print("Predictions calculated with TTA...")
            
            return self.oof_pred, self.oof_tar, self.test_pred, self.oof_fold

# Connector

def predict_fold(efficient_net, validation_img_dir, test_img_dir, model_weight_path, image_size=512):
    
    test_pred = np.empty((5, len(sample)))
    test = []
    oof_pred = []
    oof_tar = []
    oof_names = []
    oof_folds = []
    
    skf = KFold(n_splits=5,shuffle=True,random_state=42)
    for fold_number,(_,idxV) in enumerate(skf.split(np.arange(15))):
        
        print("Current Fold : {}".format(fold_number))
        
        X_val_id = df_3.loc[(df_3.tfrecord.isin(idxV)) & (df_3.patient_id!=-1)].index.values
        X_val_label = df_3.loc[(df_3.tfrecord.isin(idxV)) & (df_3.patient_id!=-1)].target.values

        valid_gen = DatasetRetriever(
            image_ids=X_val_id,
            labels=X_val_label,
            validation_img_dir=validation_img_dir,
            test_img_dir = None,
            transforms=train_augmentaions(image_size=image_size),
        )

        val_img_len = valid_gen.__len__()

        test_gen = DatasetRetriever(
            image_ids=sample.index.values,
            validation_img_dir = None,
            test_img_dir=test_img_dir,
            transforms=train_augmentaions(image_size=image_size),
        )

        validation_loader = torch.utils.data.DataLoader(
            valid_gen, 
            batch_size=32,
            num_workers=4,
            shuffle=False,
            sampler=SequentialSampler(valid_gen),
            pin_memory=False,
        )

        test_loader = torch.utils.data.DataLoader(
            test_gen, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4,
            sampler=SequentialSampler(test_gen)
        )

        model = get_model(efficient_net, fold_number, image_size, model_weight_path)

        prediction = Prediction(model = model, val_img_len = val_img_len)

        fold_oof_pred, fold_oof_tar, fold_test_pred, folds = prediction.predict(validation_loader, test_loader, fold_number)
        
        oof_pred.append(fold_oof_pred)
        oof_tar.append(fold_oof_tar)
        oof_names.append(X_val_id)
        oof_folds.append(folds)
        test_pred[fold_number, : ] = fold_test_pred
    
    test[:] = test_pred.mean(axis=0)
    
    oof = np.concatenate(oof_pred)
    true = np.concatenate(oof_tar)
    names = np.concatenate(oof_names)
    folds_total = np.concatenate([oof_folds])
    auc = roc_auc_score(true,oof)
    
    return (oof, true, names, auc, folds_total, test)

# File Generation

def make_prediction_and_save(model_name='b5', image_size=512, validation_img_dir='./', test_img_dir='./', model_weight_path='./'):
        
    oof, true, names, auc, folds_total, test = predict_fold(efficient_net=model_name, validation_img_dir=validation_img_dir, test_img_dir=test_img_dir, model_weight_path=model_weight_path, image_size=image_size)
    
    print('Overall OOF AUC with TTA = %.3f'%auc)

    a=[]
    for i in range(len(folds_total)):
        a.extend(folds_total[i])

    sample = pd.read_csv('./sample_submission.csv')
    sample.target=test

    df_oof = pd.DataFrame(dict(
        image_name = names, target=true, pred = oof, folds=a))

    df_oof.to_csv(f'{"./"}/{model_name}_{image_size}_oof.csv',index=False)
    sample.to_csv(f'{"./"}/{model_name}_{image_size}_test.csv',index=False)

make_prediction_and_save(model_name='b5', image_size=512)