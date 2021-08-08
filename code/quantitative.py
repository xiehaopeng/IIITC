'''
计算预测稳定视频的三个指标：剪裁率，失真度，稳定性得分
'''
import cv2
import numpy as np
import torch
import torch_dct as dct # pip install torch-dct
from SURF_Random import SURF_RAN_Match  # 随机采样SURF特征匹配



def CropRatio(homo):
    '''手动剪裁计算面积比率
    '''
    return 0


def Distortion(homo):
    '''输入原始抖动帧到预测稳定帧之间的单应矩阵(numframes,3,3),输入为tensor
    '''
    numFrames = homo.shape[0]
    distortions = torch.zeros(numFrames)
    for i in range(numFrames):
        A = homo[i,0:2,0:2]
        eigenValues, _ = torch.eig(A)
        distortions[i] = torch.min(eigenValues[:,0]) / torch.max(eigenValues[:,0])
    disValue = torch.min(distortions)

    return disValue



def Stability(homo):
    '''输入视频中相邻帧之间的单应矩阵(numframes-1,3,3),输入为tensor
    '''
    num = homo.shape[0]
    # 求捆绑路径
    for i in range(1,num):
        homo[i,:,:] = torch.mm(homo[i-1,:,:],homo[i,:,:])
    # 从路径中提取分量
    tx = homo[:,0,2]
    ty = homo[:,1,2]
    # DCT变换
    DCTx = dct.dct(tx)
    DCTy = dct.dct(ty)
    # 计算2～6个分量的能量比例
    DCTxValue = torch.sum(torch.pow(DCTx[1:6],2))/torch.sum(torch.pow(DCTx[1:],2))
    DCTyValue = torch.sum(torch.pow(DCTy[1:6],2))/torch.sum(torch.pow(DCTy[1:],2))

    return torch.min(DCTxValue,DCTyValue)


def getQuantitativeScore(video1, video2):
    '''输入预测稳定视频video1(已完成手动裁剪)和原始抖动视频video2两个视频路径
       输出稳定性得分和失真度
    '''
    # 读视频
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    distortionHomos = []    # 保存抖动视频到稳定视频之间的单应矩阵
    stabilityHomos = []     # 保存稳定视频相邻帧之间的单应矩阵
    frame_count = 0
    lastframe = []
    while(True):
        ret1, frame1 = cap1.read()
        if ret1 is False:
            break
        ret2, frame2 = cap2.read()
        if ret2 is False:
            break
        frame_count += 1
        print(frame_count)
        # 调用匹配算法获得抖动视频到稳定视频的单应变换
        homo,_,_ = SURF_RAN_Match(frame2,frame1)
        homo = torch.from_numpy(homo)
        distortionHomos.append(homo)
        if frame_count > 1:
            homo,_,_ = SURF_RAN_Match(lastframe,frame1)
            homo = torch.from_numpy(homo)
            stabilityHomos.append(homo)
        lastframe = frame1  # 保存前一个稳定帧
    # 将输出合并成一个tensor
    distortionHomos = torch.stack(distortionHomos, dim=0)
    stabilityHomos = torch.stack(stabilityHomos, dim=0)
    distortionval = Distortion(distortionHomos) # 失真度
    stabilityval = Stability(stabilityHomos)    # 稳定性得分
    print('失真度：',distortionval.item())
    print('稳定性得分：',stabilityval.item())



        


# test code
if __name__ == '__main__':
    # # 测试Distortion和Stability函数
    # Homo1 = torch.Tensor([1,0,0, 0,1,0, 0,0,1]).view(3,3)
    # Homo2 = torch.Tensor([1,0.1,0, 0.1,1,0, 0,0,1]).view(3,3)
    # Homo3 = torch.Tensor([2,0,0, 0,1,0, 0,0,1]).view(3,3)
    # Homo4 = torch.Tensor([1,0,50, 0,1,100, 0,0,1]).view(3,3)
    # Homos = torch.stack([Homo1, Homo2, Homo3, Homo4, Homo4, Homo4, Homo4, Homo4, Homo4, Homo4], dim=0)
    # # distortion = Distortion(Homos)
    # # print(distortion)
    # stability = Stability(Homos)
    # print(stability)
    # stabvideo = '/Users/xhpzww/Documents/gitstore/CMStabNet/result/mydata/0/0_cut2.mp4'
    stabvideo = '/Users/xhpzww/Documents/gitstore/CMStabNet/result/Regular/0/0_cut.mp4'
    orgvideo = '/Users/xhpzww/Documents/gitstore/CMStabNet/testdata/Regular/0.avi'
    getQuantitativeScore(stabvideo,orgvideo)
