from torchvision import  transforms
from albumentations.pytorch import ToTensor
import albumentations as A
import numpy as np
import  cv2
from vision.utils import Helper

class TorchTransforms():

    def __init__(self, test_transforms_list, train_transforms_list= None):
        self.train_transforms_list = train_transforms_list
        self.test_transforms_list = test_transforms_list


    def trainTransform(self):
        return transforms.Compose(self.train_transforms_list)


    def testTransform(self):
        return transforms.Compose(self.test_transforms_list)


class album_transforms():

    def __init__(self):
        helper = Helper()
        self.mean, self.std = helper.get_mean_and_std('cifar10')
        # self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        
        self.albumentation_transforms = A.Compose([
          A.PadIfNeeded(36,36),
          A.RandomCrop(32,32),
          A.Flip(),  
          A.Cutout(1,8,8,self.mean.mean()),
            A.Normalize(
                mean=self.mean,
                std=self.std
            ), ToTensor()
            ])


    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img