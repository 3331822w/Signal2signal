import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset.dataset_Unet import dataload
from torch.autograd import Variable
from model.Unet import Unet
from model.AE import AE
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import os
from tifffile import imread, imwrite
import numpy as np
import sys
from pathlib import Path
import cv2
import pylab
import matplotlib.pyplot as plt
import os
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

def cut_hypimg(img):
    # img = img[:,2:]
    img = img.T
    wave = img[:, :1]
    print(wave)
    img = img[:, 1:].reshape(1600, -1, 34)
    img = img[:, :, :].reshape(1600, -1) #cut_size
    img = np.concatenate((wave, img), axis=1)
    return img

# Parameters setting
modellr = 0.001
BATCH_SIZE = 1
EPOCHS = 20
DEVICE = torch.device('cuda')
nanophoto_img = np.loadtxt('G:/OneDrive/2022/EfficientNet/data/train/0.1s.txt')#usecols=np.arange(0,1600)
# nanophoto_img = cut_hypimg(nanophoto_img)
dataset_train = dataload('data/train', train=True, nanophoto_img=nanophoto_img)

# load data
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
criterion = nn.MSELoss()#Loss
model = Unet()#model
# model = torch.load("model_Unet.pth")
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=modellr, weight_decay=2)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 10))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    # return data/np.max(data)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    sum_num = 0
    temp_datas = []
    for i in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            sum_num = sum_num+1
            data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])#data_type: batch_size x embedding_size x text_len
            # print(data.shape)
            data = torch.as_tensor(data, dtype=torch.float32)
            data = data.permute(0, 1, 2, 3)
            # print(data.shape)
            target = target.reshape(1, target.shape[0], target.shape[1], target.shape[2])  # data_type: batch_size x embedding_size x text_len
            target = torch.as_tensor(target, dtype=torch.float32)
            target = target.permute(0, 1, 2, 3)
            data, target = Variable(data).to(device), Variable(target).to(device)
            # if i > 0:
            #     # print(i)
            #     trytry = output.detach()
            #     temp_datas.append(list(trytry.cpu().numpy()[0, 0, :, :]))
            #     seleted_number = random.randint(0,  np.array(temp_datas).shape[0]-1)
            #     random_try = np.array(temp_datas[seleted_number])
            #     random_try = random_try.reshape(1, 1, random_try.shape[0], random_try.shape[1])  # data_type: batch_size x embedding_size x text_len
            #     random_try = torch.as_tensor(random_try, dtype=torch.float32)
            #     # random_try = random_try.permute(1, 0, 2, 3)
            #     random_try = Variable(random_try).to(device)
            #     output = model(random_try)
            #     # output = model(trytry)
            # else:
            output = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
    ave_loss = sum_loss / sum_num
    print('epoch:{},loss:{}, enhance_num:{}'.format(epoch, ave_loss, sum_num))

def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
            data = torch.as_tensor(data, dtype=torch.float32)
            data = data.permute(0, 1, 2, 3)
            target = target.reshape(1, target.shape[0], target.shape[1], target.shape[2])  # data_type: batch_size x embedding_size x text_len
            target = torch.as_tensor(target, dtype=torch.float32)
            # print(target.shape)
            target = target.permute(0, 1, 2, 3)
            data, target = Variable(data).to(device), Variable(target).to(device)
            # print(data.shape)
            output = model(data)
            #############################验证结果##################################
            plt.imshow(data.cpu().numpy()[0, 0, :, :], cmap=plt.cm.jet)  # Color
            plt.title('Raman image before denoising')
            plt.savefig('./non_denoised'+'+epoch'+str(epoch)+'.jpg')
            # plt.show()
            plt.imshow(output.cpu().numpy()[0, 0, :, :], cmap=plt.cm.jet)  # Color
            plt.title('Raman image after denoising')
            plt.savefig('./denoised' + '+epoch'+str(epoch) + '.jpg')
            # plt.show()
            plt.imshow(target.cpu().numpy()[0, 0, :, :], cmap=plt.cm.jet)  # Color
            plt.title('Target Raman image')
            plt.savefig('./target' + '+epoch'+str(epoch) + '.jpg')
            # plt.show()
            loss = criterion(output, target)
            print_loss = loss.data.item()
            test_loss += print_loss
        # print('PSNR=', psnr(raw_img, output.cpu().numpy()[0, :, :, :]))
        # print('raw_PSNR=', psnr(raw_img, data.cpu().numpy()[0, :, :, :]))
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}\n'.format(
            avgloss, len(test_loader.dataset)))
# Running epochs
start = time.time()
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    val(model, DEVICE, train_loader)
torch.save(model, 'model_Unet.pth')
end = time.time()
print('time=',end-start)
