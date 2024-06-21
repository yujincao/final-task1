import torch
from torch import nn
import numpy as np
from torchvision.transforms import transforms
from torchvision import transforms, datasets

np.random.seed(0)


class GaussianBlur(object):
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class ViewGen(object):
    # Take 2 random crops
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class GetTransformedDataset:
    @staticmethod
    def get_simclr_transform(size, s=1):
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply(
                                                  [color_jitter], p=0.8),
                                              transforms.RandomGrayscale(
                                                  p=0.2),
                                              GaussianBlur(
                                                  kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_cifar10_train(self, n_views):
        return datasets.CIFAR10('./data', train=True,
                                transform=ViewGen(
                                    self.get_simclr_transform(
                                        32),
                                    n_views),
                                download=True)
    
    def get_cifar10_test(self, n_views):
        return datasets.CIFAR10('./data', train=False,
                                transform=transforms.ToTensor(),
                                download=True)
    
    def get_cifar100_train(self):
        return datasets.CIFAR100(root='./data', 
                                 train=True, 
                                 download=True, 
                                 transform=transforms.ToTensor())

    def get_cifar100_test(self):
        return datasets.CIFAR100('./data', train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)