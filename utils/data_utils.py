import torch
import numpy as np


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0],1,-1)
    elif len(X.shape) == 3:
        return X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
    else:
        return 'dimensional error'