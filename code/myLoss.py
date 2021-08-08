import cv2
import time
import numpy as np
import torch 
import torch.nn as nn 
from Homography import HomographyFunc_P,HomographyFunc_I, convert_image_np # 对点、图片的扭曲函数
from SURF_Random import SURF_RAN_Match  # 随机采样SURF特征匹配
# from extract_feature import extract_VGGfeature  # VGG特征提取
from LKflow_warp import compute_TVL1flow       # tvl1光流扭曲函数


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
imgnum = 0


class Pixel_LossFunc(nn.Module): 
    """像素损失"""
    
    def __init__(self): 
        super(Pixel_LossFunc, self).__init__() 


    def forward(self, FItxs, labels): 
        '''输入为扭曲后的不稳定帧FItxs(S, B, C, H, W)，地面真值稳定帧labels(S, B, C, H, W) 都是经过归一化标准化预处理'''
        lfn = nn.MSELoss()
        loss = lfn(FItxs,labels)
        return loss


class Feature_LossFunc(nn.Module): 
    """特征损失(特征匹配失败则返回一个很小的loss)
    """
    def __init__(self): 
        super(Feature_LossFunc, self).__init__() 


    def forward(self, Ft, Itx, Ity): 
        '''输入为扭曲参数Ft,原始不稳定帧Itx(C,H,W)，地面真值稳定帧Ity(C,H,W)
        '''
        # 首先要把tensor,float32(C,H,W)的Ix和Iy 改成numpy,uint8(H,W,C)的（0-255）图片格式
        Itx = np.uint8(Itx.numpy().transpose(1,2,0))
        Ity = np.uint8(Ity.numpy().transpose(1,2,0))
        # 调用匹配算法获得两个匹配点列表
        _,pA,pB = SURF_RAN_Match(Itx,Ity)
        if len(pA): # 匹配成功
            pA = np.array(pA)
            pB = np.array(pB)
            # 经过单应映射
            pA = HomographyFunc_P(Ft.cpu(),torch.Tensor(pA))
            lfn = nn.MSELoss()
            loss = lfn(torch.Tensor(pB),pA)
        else:   # 匹配失败则指定一个极小的loss
            loss = torch.Tensor(np.array(1e-8))
            loss.requires_grad = True
        return loss.to(dev)/(255*255)


# class Vgg_LossFunc(nn.Module): 
#     """VGG损失
#     """
#     def __init__(self): 
#         super(Vgg_LossFunc, self).__init__() 
#     def forward(self, FItxs, labels): 
#         '''输入为扭曲后的不稳定帧FItxs(S, B, C, H, W)，地面真值稳定帧labels(S, B, C, H, W) 都是经过归一化标准化预处理'''
#         l_f=nn.L1Loss()
#         # VGG网络输入需要是(B,C,H,W)，需要处理一下维度
#         feature_x, feature_y = extract_VGGfeature(FItxs.view(-1,3,256,256), labels.view(-1,3,256,256))
#         loss = l_f(feature_x, feature_y)
#         return loss


class Temporal_LossFunc(nn.Module): 
    """时间损失
    """
    def __init__(self): 
        super(Temporal_LossFunc, self).__init__() 


    def forward(self, FItxs): 
        '''输入为扭曲后的不稳定帧FItxs(S, B, C, H, W),都是归一化后的数据
        '''
        # # 反归一化来计算TVL1光流
        # FIt_1x = compute_TVL1flow(convert_image_np(FIt_1x),convert_image_np(FItx))

        lfn = nn.MSELoss()
        loss = lfn(FItxs[1:,...], FItxs[0:FItxs.shape[0]-1,...].detach())    # .detach()
        return loss


class outL_Pixel_LossFunc(nn.Module): 
    """outL的像素损失+vgg损失
    """
    def __init__(self): 
        super(outL_Pixel_LossFunc, self).__init__() 


    def forward(self, outL, labels): 
        '''输入为解码器预测的outL(S, B, C, H, W)，地面真值稳定帧labels(S, B, C, H, W),都是归一化后的数据
        '''
        lfn = nn.MSELoss()
        loss = lfn(outL, labels)
        return loss


# class outL_Vgg_LossFunc(nn.Module): 
#     """outL的vgg损失
#     """
#     def __init__(self): 
#         super(outL_Vgg_LossFunc, self).__init__() 


#     def forward(self, outL, labels): 
#         '''输入为扭曲后的不稳定帧FItxs(S, B, C, H, W)，地面真值稳定帧labels(S, B, C, H, W) 都是经过归一化标准化预处理'''
#         l_f=nn.L1Loss()
#         # VGG网络输入需要是(B,C,H,W)，需要处理一下维度
#         feature_x, feature_y = extract_VGGfeature(outL.view(-1,3,256,256), labels.view(-1,3,256,256))
#         loss = l_f(feature_x, feature_y)
#         return loss


class Stab_LossFunc(nn.Module): 
    def __init__(self, step_time, batch_size): 
        super(Stab_LossFunc, self).__init__()

        self.step_time = step_time
        self.batch_size = batch_size
        # 准备各损失函数
        self.lossFunc_P = Pixel_LossFunc()
        self.lossFunc_F = Feature_LossFunc()
        # self.lossFunc_V = Vgg_LossFunc()
        self.lossFunc_T = Temporal_LossFunc()
        self.lossFunc_LP = outL_Pixel_LossFunc()
        # self.lossFunc_LV = outL_Vgg_LossFunc()
        


    def forward(self, out_Y, out_L, inputs, Ix, labels, Iy): 
        '''计算四个损失的加权和
           注意输入的out_Y, out_L, inputs, labels都是cuda ; Ix, Iy是在cpu中
           如果后面用到opencv包中的函数注意格式转换
        '''
        loss_P=0
        loss_F=0
        loss_V=0
        loss_T=0
        loss_LP=0
        loss_LV=0
        global imgnum

        # 应用单应扭曲获得扭曲后的不稳定帧FItxs，用于求出像素损失
        FItxs = HomographyFunc_I(out_Y, inputs)


        # 中间过程可视化
        orginput = convert_image_np(inputs[4,0,...]) # 原始不稳定
        out = convert_image_np(FItxs[4,0,...])       # 预测扭曲后的图像
        labout = convert_image_np(labels[4,0,...])   # 地面真值标签
        outL = convert_image_np(out_L[4,0,...])      # outL

        midnum = int(inputs.shape[0]/2 + 1)
        orginputmid = convert_image_np(inputs[midnum,0,...]) # 原始不稳定
        outmid = convert_image_np(FItxs[midnum,0,...])       # 预测扭曲后的图像
        laboutmid = convert_image_np(labels[midnum,0,...])   # 地面真值标签
        outLmid = convert_image_np(out_L[midnum,0,...])      # outL

        lastnum = int(inputs.shape[0] - 1)
        orginputlast = convert_image_np(inputs[lastnum,0,...]) # 原始不稳定
        outlast = convert_image_np(FItxs[lastnum,0,...])       # 预测扭曲后的图像
        laboutlast = convert_image_np(labels[lastnum,0,...])   # 地面真值标签
        outLlast = convert_image_np(out_L[lastnum,0,...])      # outL
        # 组合在一起
        htich1 = np.hstack((orginput,labout,out,outL))
        htich2 = np.hstack((orginputmid,laboutmid,outmid,outLmid))
        htich3 = np.hstack((orginputlast,laboutlast,outlast,outLlast))
        vtich = np.vstack((htich1, htich2, htich3))
        cv2.imwrite("./img/res/result_img_"+str(imgnum)+".jpg", vtich)
        imgnum = imgnum + 1
        if imgnum > 2:
            imgnum = 0


        # 计算各个损失
        for batchid in range(self.batch_size):
            for stepid in range(self.step_time):
                # 计算损失特征
                loss_F += self.lossFunc_F(torch.squeeze(out_Y[stepid,batchid,...]), Ix[stepid,batchid,...], Iy[stepid,batchid,...])

        loss_P = self.lossFunc_P(FItxs, labels)
        loss_F = loss_F/(self.step_time*self.batch_size)
        # loss_V = self.lossFunc_V(FItxs, labels)
        loss_LP = self.lossFunc_LP(out_L, labels)
        # loss_LV = self.lossFunc_LV(out_L, labels)
        loss_T = self.lossFunc_T(FItxs)

        # 各损失乘以权重
        loss_P = loss_P*1
        loss_F = loss_F*500
        loss_V = loss_V*1
        loss_T = loss_T*1
        loss_LP = loss_LP*1
        loss_LV = loss_LV*1

        print("像素损失= " + str(format(loss_P.item(), '.2f')), end=' ')
        print("特征损失= " + str(format(loss_F.item(), '.2f')), end=' ')
        # print("VGG损失= " + str(format(loss_V.item(), '.2f')), end=' ')
        print("时间损失 = " + str(format(loss_T.item(), '.2f')), end=' ')
        print("LP损失= " + str(format(loss_LP.item(), '.2f')), end=' ')
        # print("LV损失= " + str(format(loss_LV.item(), '.2f')), end=' ')
        return loss_P, loss_F, loss_V, loss_T, loss_LP, loss_LV


if __name__ == '__main__':

    # 批数量8和时间步20
    batch_size = 2
    step_time = 4
    # 原始不稳定帧
    Ix = torch.rand(step_time,batch_size,3,256,256)
    # 地面真值帧
    Iy = torch.rand(step_time,batch_size,3,256,256)

    out_Y = torch.rand(step_time,batch_size,8,1,1)

    criterion = Stab_LossFunc(step_time, batch_size)
    loss = criterion(out_Y,Ix,Iy)
    print(loss)

