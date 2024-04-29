import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset.dataset import DogCat
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from model.Unet import Unet
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
import matplotlib.pyplot as plt
from scipy import ndimage
from model.EfficientNetV2 import efficientnetv2_s
from model.resnet import resnet18, resnet34, resnet50, resnet101
from model.ALEXNET import vgg16
from torch.nn import functional as F

# 设置全局参数
modellr = 1e-3
BATCH_SIZE = 36
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cut_hypimg(img):
    img =img.T
    wave = img[:, :1]
    img = img[:, 1:].reshape(1600, -1, 20)
    img = img[:, :, :].reshape(1600, -1) #裁切图像中的谱图向量,列*行
    img = np.concatenate((wave, img), axis=1)
    return img


# 读取数据
nanophoto_img = np.loadtxt('G:/OneDrive/2022/EfficientNet/data/train/0.1s.txt')#,usecols=np.arange(0,1600)
# nanophoto_img = cut_hypimg(nanophoto_img)
dataset_test = dataload('data/train',nanophoto_img=nanophoto_img, test=True)
# 导入数据
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()#交叉熵
model = torch.load("model_Unet.pth")
# 对应文件夹的label.
model.to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=modellr)#求解器

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def distance_Reciprocal(a,b):
    d_all=a**2+b**2
    d=d_all**0.5
    d=1/d
    return

def weight_result(data,c):
    m = (c-1)//2
    n=len(data)
    new_data= [0 for x in range(n)]
    for i in range(0,n):
        w_sum=0
        if i <= m - 1:#左端不完整的几个点
            lx = 0
            ly = i + m
            p_sum=0
            w_sum=w_sum+data[0]*(m-i)
            a=0
            happens = [0 for x in range(ly-lx+1)]
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range(lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
        elif i >= n - m:#右端不完整的几个点
            lx = i - m
            ly = n-1
            p_sum=0
            w_sum=w_sum+data[n-1]*(m-n+i)
            a=0
            happens = [0 for x in range(ly-lx+1)]
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range (lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
        else:
            lx = i - m
            ly = i + m
            p_sum=0
            happens = [0 for x in range(ly-lx+1)]
            a=0
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range (lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
    return new_data

def bad_point_remove(img):
    for i in range(5):
        img = normalization(img)
        index_max = np.argmax(img)
        index_min = np.argmin(img)
        img[index_max] = weight_result(img, 3)[index_max]
        img[index_min] = weight_result(img, 3)[index_min]
    return img

#测试
def test(model, device, test_loader):
    before_denoising_target=[]
    before_denoising_nontarget=[]
    model.eval()#套路
    correct = 0
    test_loss = 0
    total_num = len(test_loader.dataset)
    denoised_all = np.zeros((34, 50))
    nondenoised_all = np.zeros((34, 50))
    with torch.no_grad():
        for i in range(100):
            for data, target in test_loader:
                # before_denoising_target.append(float(np.array(data[0, 2, 5])))
                # before_denoising_nontarget.append(float(np.array(data[0, 30, 0])))
                data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])  # data_type: batch_size x embedding_size x text_len
                data = torch.as_tensor(data, dtype=torch.float32)
                data = data.permute(1, 0, 2, 3)
                target = target.reshape(1, target.shape[0], target.shape[1], target.shape[2])  # data_type: batch_size x embedding_size x text_len
                target = torch.as_tensor(target, dtype=torch.float32)
                target = target.permute(1, 0, 2, 3)
                data, target = Variable(data).to(device), Variable(target).to(device)
                output = model(target)
                denoised_all = denoised_all + output.cpu().numpy()[0, 0, :, :]
                nondenoised_all = nondenoised_all + data.cpu().numpy()[0, 0, :, :]
                # plt.matshow(output.cpu().numpy()[0, 0, :, :], cmap=plt.cm.jet)  # 这里设置颜色为红色，也可以设置其他颜色
                # plt.title('Raman image after denoising')
                # plt.savefig('./denoised' + '.jpg')
                # # plt.show()
                # plt.matshow(target.cpu().numpy()[0, 0, :, :], cmap=plt.cm.jet)  # 这里设置颜色为红色，也可以设置其他颜色
                # plt.title('Target Raman image')
                # plt.savefig('./target' + '.jpg')
                # plt.show()
                loss = criterion(output, target)
                print_loss = loss.data.item()
                test_loss += print_loss
                print(i)
        # #############################remove bad points#####################################
        # nondenoised_all = nondenoised_all.ravel()
        # denoised_all = denoised_all.ravel()
        # nondenoised_all = bad_point_remove(nondenoised_all)
        # denoised_all = bad_point_remove(denoised_all)
        # nondenoised_all = np.array(nondenoised_all).reshape((int(len(nondenoised_all) / 30)), 30)
        # denoised_all = np.array(denoised_all).reshape((int(len(denoised_all) / 30)), 30)
        # ####################################################################################
        denoised_all = normalization(denoised_all)
        nondenoised_all = normalization(nondenoised_all)
        # print('denoised_target=', denoised_all[1, 5])
        # print('nondenoised_target=', nondenoised_all[1, 5])
        # print('denoised_nontarget=', denoised_all[29, 0])
        # print('nondenoised_target=', nondenoised_all[29, 0])
        # print('float_y:', before_denoising_target)
        # print('float_n:', before_denoising_nontarget)
        plt.matshow(nondenoised_all, cmap=plt.cm.jet)  # 这里设置颜色为红色，也可以设置其他颜色
        plt.title('Raman image')
        plt.savefig('./nondenoised' + '.jpg')
        plt.show()
        np.savetxt('./nondenoised.txt', nondenoised_all)
        nondenoised_all = ndimage.median_filter(nondenoised_all, (3, 3))
        plt.matshow(nondenoised_all, cmap=plt.cm.jet)  # 这里设置颜色为红色，也可以设置其他颜色
        plt.title('Raman image')
        plt.savefig('./nondenoised+media' + '.jpg')
        plt.show()
        np.savetxt('./nondenoised+media.txt', nondenoised_all)
        plt.matshow(denoised_all, cmap=plt.cm.jet)   # 这里设置颜色为红色，也可以设置其他颜色
        plt.title('Raman image')
        plt.savefig('./denoised' + '.jpg')
        plt.show()
        np.savetxt('./denoised.txt', denoised_all)
        denoised_all = ndimage.median_filter(denoised_all, (3, 3))
        plt.matshow(denoised_all, cmap=plt.cm.jet)  # 这里设置颜色为红色，也可以设置其他颜色
        plt.title('Raman image')
        plt.savefig('./denoised+media' + '.jpg')
        plt.show()
        np.savetxt('./denoised+media.txt', denoised_all)
        avgloss = test_loss / len(test_loader)
        print('\ntest set: Average loss: {:.4f}'.format(avgloss))

test(model, DEVICE, test_loader)
