"""pip uninstall opencv-python
   pip install opencv-contrib-python或者pip install opencv-python
"""
import torch 
import torch.nn as nn 
import torchvision.models as models


# cuda
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 准备vgg19网络
vgg19 = models.vgg19(pretrained=True).features
vgg19 = vgg19.to(dev).eval()  # 一定要有这行，不然运算速度会变慢,并且会影响特征提取结果
# vgg19.eval()


# 特征提取
def extract_VGGfeature(img1, img2):
   result1=vgg19(img1)
   result2=vgg19(img2)
   return result1, result2	# 返回的矩阵shape是(B, C, H, W)