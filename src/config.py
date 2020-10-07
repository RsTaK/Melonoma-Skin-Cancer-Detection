import torch

class Config:
    num_workers=2
    batch_size=4
    n_epoches=100
    lr = 0.00003
    patience=5
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='max',
        factor=0.8,
        patience=1,
        verbose=True, 
        threshold=0.00001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )