import torch
x = torch.rand(5,3)
print("Is there a CUDA?",torch.cuda.is_available())
print('No. of GPU:',torch.cuda.device_count())