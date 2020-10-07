import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_augmentaions(size):
    return A.Compose([
            A.Resize(height=size, width=size, p=1),
            A.RandomSizedCrop(min_max_height=(size-(size*0.2), size-(size*0.2)), height=size, width=size, p=0.5), # 20% of height and width to be reduced
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(max_holes=8, max_width=64, max_height=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),                  
        ], p=1.0)

def validation_augmentations(size):
    return A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)