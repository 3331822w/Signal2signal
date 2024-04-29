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

# Parameters setting
modellr = 1e-3
BATCH_SIZE = 36
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cut_hypimg(img):
    img =img.T
    wave = img[:, :1]
    img = img[:, 1:].reshape(1600, -1, 20)
    img = img[:, :, :].reshape(1600, -1) #size
    img = np.concatenate((wave, img), axis=1)
    return img


nanophoto_img = np.loadtxt('G:/OneDrive/2022/EfficientNet/data/train/0.1s.txt')#,usecols=np.arange(0,1600)
dataset_test = dataload('data/train',nanophoto_img=nanophoto_img, test=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()
model = torch.load("model_Unet.pth")
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=modellr)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


#测试
def test(model, device, test_loader):
    before_denoising_target=[]
    before_denoising_nontarget=[]
    model.eval()
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
                loss = criterion(output, target)
                print_loss = loss.data.item()
                test_loss += print_loss
                print(i)
        denoised_all = normalization(denoised_all)
        nondenoised_all = normalization(nondenoised_all)
        plt.matshow(nondenoised_all, cmap=plt.cm.jet) 
        plt.title('Raman image')
        plt.savefig('./nondenoised' + '.jpg')
        plt.show()
        np.savetxt('./nondenoised.txt', nondenoised_all)
        nondenoised_all = ndimage.median_filter(nondenoised_all, (3, 3))
        plt.matshow(nondenoised_all, cmap=plt.cm.jet)  
        plt.title('Raman image')
        plt.savefig('./nondenoised+media' + '.jpg')
        plt.show()
        np.savetxt('./nondenoised+media.txt', nondenoised_all)
        plt.matshow(denoised_all, cmap=plt.cm.jet) 
        plt.title('Raman image')
        plt.savefig('./denoised' + '.jpg')
        plt.show()
        np.savetxt('./denoised.txt', denoised_all)
        denoised_all = ndimage.median_filter(denoised_all, (3, 3))
        plt.matshow(denoised_all, cmap=plt.cm.jet) 
        plt.title('Raman image')
        plt.savefig('./denoised+media' + '.jpg')
        plt.show()
        np.savetxt('./denoised+media.txt', denoised_all)
        avgloss = test_loss / len(test_loader)
        print('\ntest set: Average loss: {:.4f}'.format(avgloss))

test(model, DEVICE, test_loader)
