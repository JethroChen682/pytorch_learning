import torch
net1=torch.load("nnnnnn1.pth")
net2=torch.load("nn2.pth")
d=torch.device("cuda:0")
cc1=net1(torch.Tensor([1]).to(d))#2
cc2=net2(torch.Tensor([12]).to(d))#[2]
print(f"cc1:{cc1}\ncc2:{cc2}")