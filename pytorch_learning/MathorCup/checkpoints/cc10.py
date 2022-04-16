import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from datetime import datetime
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data[0]
    return num_correct / total
def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data[0]
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一层是 5x5 的卷积, 输入的 channels 是 3，输出的 channels 是 64, 步长是 1, 没有 padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(True))

        # 第二层是 3x3 的池化, 步长是 2, 没有 padding
        self.max_pool1 = nn.MaxPool2d(3, 2)

        # 第三层是 5x5 的卷积, 输入的 channels 是 64，输出的 channels 是 64, 步长是 1, 没有 padding
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True))

        # 第四层是 3x3 的池化，步长是 2，没有 padding
        self.max_pool2 = nn.MaxPool2d(3, 2)

        # 第五层是全连接层，输入是 1204，输出是 384
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 384),
            nn.ReLU(True))#True 表示原地操作

        # 第六层是全连接层，输入是 384，输出是 192
        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(True))

        # 第七层是全连接层，输入是 192，输出是 10
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        # 将矩阵拉平
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x