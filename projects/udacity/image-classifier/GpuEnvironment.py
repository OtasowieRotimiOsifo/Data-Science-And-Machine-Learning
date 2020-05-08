import torch

class GpuEnvironment:
    def __init__(self):
        if torch.cuda.is_available():
           self.usegpu = True
           self.device = "cuda:0"
        else:
           self.usegpu = False
           self.device = "cpu"
    
    def set_use_gpu(self, use_gpu):
        use_gpu_loc = eval(use_gpu)
        if torch.cuda.is_available() and use_gpu_loc == True:
           self.usegpu = True
           self.device = "cuda:0"
        else:
           self.usegpu = False
           self.device = "cpu"
    