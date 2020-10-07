import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def Net(model_name='b0'):
    model = EfficientNet.from_pretrained(f'efficientnet-{model_name}')
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=2, bias=True)
    return model