import os

from torch.utils.data import  Dataset
from utils import *
from torchvision import  transforms
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
#将图片进行旋转和裁剪，提升模型的泛化能力

class MydataSet(Dataset):
    def __init__(self,path):
        self.path = path
        self.name = os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self,index):  #根据index获取对应的样本
        segment_name = self.name[index]  #xx.png
        segment_path = os.path.join(self.path,'SegmentationClass',segment_name) #拿标签的地址
        image_path = os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg')) #拿图像的地址
        segment_image = keep_image_size_open(segment_path, mode =  'L')#读取标签图像时，将其变为单通道
        image = keep_image_size_open(image_path)

        return  data_transforms(image), data_transforms(segment_image)

if __name__ == '__main__':
    data = MydataSet('VOCtrainval_11-May-2012/VOCdevkit/VOC2012')
    print(data[0][0])
    print(data[0][1])

