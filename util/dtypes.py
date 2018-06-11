import torch

float = torch.FloatTensor
int   = torch.IntTensor
long  = torch.LongTensor
byte  = torch.ByteTensor
ones  = torch.ones
zeros = torch.zeros

def set_gpu_dtypes():
    global float
    global int
    global long
    global byte
    global ones
    global zeros

    float = torch.cuda.FloatTensor
    int   = torch.cuda.IntTensor
    long  = torch.cuda.LongTensor
    byte  = torch.cuda.ByteTensor
    def ones(*args):  return torch.cuda.FloatTensor(*args).zero_()+1
    def zeros(*args): return torch.cuda.FloatTensor(*args).zero_()
