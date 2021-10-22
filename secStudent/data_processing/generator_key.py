import torch

key = (torch.randint(0, 255, [1, 128, 128]) -128) / 255
key = torch.cat([key, key], dim=2)
key = torch.cat([key, key], dim=2)
key = torch.cat([key, key], dim=1)
print(key.shape)
torch.save(key, "./key.pth")