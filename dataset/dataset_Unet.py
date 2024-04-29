# coding:utf8
import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from scipy.linalg import hankel
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from scipy import optimize
from scipy.stats import norm

def distance_Reciprocal(a,b):
    d_all=a**2+b**2
    d=d_all**0.5
    d=1/d
    return d

def weight_result(data,c):
    m = (c-1)//2
    n=len(data)
    new_data= [0 for x in range(n)]
    for i in range(0,n):
        w_sum=0
        if i <= m - 1:
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

def closest(mylist, Number):
    answer = []
    for i in mylist:
        answer.append(abs(Number-i))
    return answer.index(min(answer))

def located_normalization(data, g, h):
    import  numpy as np
    max1 = max(data[g:h])
    data_new = [g / max1 for g in data]
    return np.array(data_new)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def normalization2(data):
    data=data/np.max(data)
    return data


def derivative(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    result = np.array(result)
    return result

def f_1(x, A, B):
    return A * x + B

def random_transform(data, size):
    data = data.T
    data = np.flipud(data)
    wave = data[:, :1]
    wave = np.squeeze(wave)
    image = data[:, 1:]
    img = []
    row = 50#列
    start = closest(wave, 135)  # closest(wave,1000)
    end = closest(wave, 155)  # closest(wave,2000)
    target = list(range(len(wave)))
    target = target[start:end]
    target_sum = []
    for i in range(size):
        target_sum.append(random.sample(list(target[0 + i * (len(target) // size):(i + 1) * (len(target) // size)]), 1))
    for i in range(1, np.shape(data)[1]):
        values2 = []
        values = image[:, i - 1:i]
        values = np.squeeze(values)
        for m in target_sum:
            values2.append(values[m])
        peak_intensity = sum(values2)  # peak area
        img.append(peak_intensity)
    for i in range(5):
        img = normalization(img)
        index_max = np.argmax(img)
        index_min = np.argmin(img)
        img[index_max] = weight_result(img, 3)[index_max]
        img[index_min] = weight_result(img, 3)[index_min]
    img = normalization(img)
    img = np.array(img).reshape((int(len(img) / row)), row)
    return img

class dataload(data.Dataset):

    def __init__(self, root, nanophoto_img, transforms=None, train=True, test=False,):
        self.test = test
        self.transforms = transforms
        self.nanophoto_img = nanophoto_img
        if os.listdir(root)[0].endswith(".txt"):#txt data type
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            imgs=[]
            dirs = [os.path.join(root, dirs) for dirs in os.listdir(root)]
            for l in dirs:
                imgs = imgs + [os.path.join(l, imgs) for imgs in os.listdir(l)]
        for k in range(len(imgs)):
            imgs[k] = imgs[k].replace('\\','/')
        self.imgs = imgs

    def __getitem__(self, index):
        img = self.nanophoto_img
        data = random_transform(img, 20)
        target = random_transform(img, 20)
        #################try###################
        # plt.matshow(data, cmap=plt.cm.jet)  # color
        # plt.title('Raman image before denoising')
        # plt.show()
        # plt.matshow(target, cmap=plt.cm.jet)  # color
        # plt.title('Raman image after denoising')
        # plt.show()
        return data, target

    def __len__(self):
        return len(self.imgs)
