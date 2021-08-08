import cv2
import time
import torch
import numpy as np
import torch.nn.functional
from PIL import Image
from torchvision import transforms  # 预处理
import matplotlib.pyplot as plt


# 加载设备
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def changeFweight(out_Y):   # 传入函数的tensor是引用
    '''
    用来改变扭曲矩阵中个参数的权重
    out_Y(stepid,batchid,6,1,1])
    '''
    # 缩放给1，旋转参数给1权重，平移参数给10权重
    # out_Y[:,:,1,:,:] = 5*out_Y[:,:,1,:,:]
    # out_Y[:,:,3,:,:] = 5*out_Y[:,:,3,:,:]
    out_Y[:,:,2,:,:] = 10*out_Y[:,:,2,:,:]
    out_Y[:,:,5,:,:] = 10*out_Y[:,:,5,:,:]


def mytransform(frame):
    trans = transforms.Compose([
        transforms.ToTensor(),      # 将PIL.Image转化为tensor，即归一化过程,并图片格式为(C,H,W)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
    frame = trans(frame)
    return frame


def convert_image_np(inp):
    """把经过归一化和标准化预处理的tensor图片转化为opencv格式"""
    inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = inp * 255
    inp = inp.astype('uint8')
    inp = cv2.cvtColor(inp,cv2.COLOR_RGB2BGR)
    return inp


def HomographyFunc_P(F, pA):
    """求点pA的单应变换（透视变换）(tensor操作)
       input: 6参数向量F(tensor)和扭曲目标点pA(tensor)格式为(num, 2)
       output: 扭曲后的点Homography_P(tensor)(num,2)
    """
    # 单应变换矩阵
    T = torch.cat((F,torch.Tensor([0,0,1]) ),0).view(3,3)
    # T = torch.cat((F,torch.Tensor([1]) ),0).view(3,3)

    pAsize = pA.shape
    src_point = torch.cat([pA, torch.ones([pAsize[0],1]) ],dim=1)  # 变换前的点转化为齐次坐标(num,3)
    src_point_transpose = src_point.t() # (3,num)
    a = torch.mm( T, src_point_transpose )  # (3,num)
    Homography_P = torch.div(a[0:2,:], a[2,:])  # (2,num)
    return Homography_P.t() # (num,2)


def HomographyFunc_I(F, Ix):
    """用仿射变换扭曲图片(tensor操作)
       input: 6参数向量F(tensor)(S,B,6,1,1)和经过预处理的输入数据Ix(tensor)格式为(S, B, C, H, W)(B*S,3,256,256)
       output: 扭曲后的目标Homography_I(tensor)(S, B, C, H, W)
    """
    step_time, batch_num, channel, height, width = Ix.shape
    Ixcopy = Ix.clone().view(-1,channel,height,width)

    part = torch.zeros([step_time,batch_num,3,1,1]).to(dev)
    part[:,:,2,...] = 1

    # 仿射变换矩阵
    T = torch.cat((F, part),2).view(batch_num*step_time,3,3)

    # 将仿射变换矩阵转换为pytorch中affine_grid的theta
    T = torch.inverse(T)
    theta = torch.zeros([Ixcopy.shape[0],2,3]).to(dev)
    theta[:,0,0] = T[:,0,0]
    theta[:,0,1] = T[:,0,1]*height/width
    theta[:,0,2] = T[:,0,2]*2/width + theta[:,0,0] + theta[:,0,1] - 1
    theta[:,1,0] = T[:,1,0]*width/height
    theta[:,1,1] = T[:,1,1]
    theta[:,1,2] = T[:,1,2]*2/height + theta[:,1,0] + theta[:,1,1] - 1

    # theta = torch.unsqueeze(theta,dim=0)
    # Ix = torch.unsqueeze(Ix,dim=0)
    grid = torch.nn.functional.affine_grid(theta, Ixcopy.size())
    # Homography_I = torch.nn.functional.grid_sample(Ix, grid, mode='bilinear', padding_mode="zeros", align_corners=True)
    Homography_I = torch.nn.functional.grid_sample(Ixcopy, grid)

    return Homography_I.view(step_time, batch_num, channel, height, width)


def HomographyFunc(F, Ix):
    """用opencv封装函数求单应变换
       input: 6参数向量F(tensor)和扭曲目标Ix(tensor)格式为(C, H, W)
       output: 扭曲后的目标Homography_I(tensor)
       虽然很快但是好像无法传递梯度信息
    """
    channel, height, width = Ix.shape
    # 转化为opencv的格式
    Ix = Ix.numpy().transpose(1,2,0)
    # 单应变换矩阵
    T = torch.cat((F,torch.Tensor([0,0,1]) ),0).view(3,3)
    T = T.detach().numpy()

    # 变换后图像（注意还是opencv格式h,w,c,需要转化一下)
    Homography_I = cv2.warpPerspective(Ix,T,(width,height))
    Homography_I = torch.Tensor(Homography_I.transpose(2,0,1))
    return Homography_I


# test code
if __name__ == '__main__':
    # 变换矩阵
    F = torch.Tensor([0.466 , 0.199 , 9.291 , 0.301 , 0.317 , 26.614])


    # 测试HomographyFunc
    image = cv2.imread("img/1.jpg")
    # opencv读的图像尺寸是（H，W，C）需要改一下
    # transpose函数把对应的维度进行互换
    image = image.transpose(2,0,1)
    image = torch.Tensor(image)
    # time_start=time.time()
    print(F)
    out = HomographyFunc(F,image)
    # time_end=time.time()
    # print(time_end-time_start)
    out = out.cpu().numpy()
    out = out.transpose(1,2,0)
    # print(out)
    # print(out.shape)
    cv2.imwrite("img/out2.jpg", out)
    # # cv2.imshow("result", out)
    # # cv2.waitKey(0)

    # 测试HomographyFunc_I
    image = cv2.imread("img/1.jpg")
    # 图片预处理前需要把opencv转化为PIL Image的格式
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    image = mytransform(image)
    print(F)
    out = HomographyFunc_I(F,image)
    out = convert_image_np(out)
    # print(out.shape)
    cv2.imwrite("img/out.jpg", out)
    # cv2.imshow("result", out)
    # cv2.waitKey(0)

    # 测试HomographyFunc_P
    pA = torch.Tensor([[10,40],[50,50]])
    print(F)
    print(pA)
    print(HomographyFunc_P(F, pA))



    

