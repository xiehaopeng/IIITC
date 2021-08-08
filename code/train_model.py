"""
训练    python train_model.py
"""
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from myDataset import MyDataset     # 数据集
from torchvision import transforms  # 预处理
from torch.utils.data import DataLoader
from IIITCNET import CMStabNet     # CMStabNet循环网络
from myLoss import Stab_LossFunc    # 损失函数
import torch.optim as optim         
from torch.optim import lr_scheduler
from Homography import changeFweight


# 超参数设置
epochs = 15         # 训练总轮数 20
batch_size = 2      # 批数量 1
step_time = 40      # 序列串长度（循环网络循环次数） 24

# 数据集的预处理设置
train_transforms = transforms.Compose([
        transforms.ToTensor(),      # 将PIL.Image转化为tensor，即归一化过程,并图片格式为(C,H,W)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 创建数据集和加载器
train_dataset=MyDataset(u'../256_256','.avi', step_time,transform=train_transforms)
train_dataloder=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
# 组织设备
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建加载网络
CMNet = CMStabNet((256,256), 3, step_time, batch_size)
# CMNet.load_state_dict(torch.load('model/eyeFmodel8.pth', map_location='cpu'))
# CMNet.load_state_dict(torch.load('model/40nostart_minidata_lossPFV_2.pth')) 
CMNet = CMNet.to(dev)

# 损失函数，优化器和学习衰减策略
train_criterion = Stab_LossFunc(step_time, batch_size)
train_optimizer = optim.Adam(CMNet.parameters(), lr=0.002, betas=(0.5,0.999))           # Adam优化器
train_scheduler = lr_scheduler.StepLR(train_optimizer, step_size=5, gamma=0.1)         # 每10轮下降为0.1


since = time.time()
loss_his_P=[]
loss_his_F=[]
loss_his_V=[]
loss_his_T=[]
loss_his_LP=[]
loss_his_LV=[]
# 训练过程
for epoch in range(epochs):
    CMNet.train()
    print('-' * 40)
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    batchid = 0
    allbatch = int(len(train_dataset)/batch_size)   # 一轮的总批数
    for inputs, inputimg, labels, labelimg in train_dataloder:  # 拿批数据
        if inputs.shape[0] != batch_size:   # 最后拿出来的数据不足一批则进入下一轮
            break
        print('Epoch {}/{}  '.format(epoch, epochs - 1),end=" ")
        print('batch {}/{}  '.format(batchid, allbatch - 1),end=" ")

        # 由于拿出来的数据格式为(B,S,C,H,W)，而网络数据为(S,B,C,H,W)，要改一下
        inputs = inputs.transpose(0,1)
        labels = labels.transpose(0,1)
        inputimg = inputimg.transpose(0,1)
        labelimg = labelimg.transpose(0,1)
        inputs = inputs.to(dev)
        labels = labels.to(dev)
        # inputimg = inputimg.to(dev)
        # labelimg = labelimg.to(dev)

        # 梯度清零
        train_optimizer.zero_grad()

        # 前向传播
        # print("正在执行前向传播...",end=" ")
        # time_start=time.time()
        out_Y,out_L = CMNet(inputs) # 输出扭曲参数(S,B,1,1,8)和解码后的未剪裁稳定帧(S,B,3,256,256)
        # time_end=time.time()
        # print("总用时:" + str(format(time_end-time_start, '.2f')) + "s") 

        # 改变扭曲矩阵各参数权重
        changeFweight(out_Y)


        # 计算损失
        loss_P, loss_F, loss_V, loss_T, loss_LP, loss_LV = train_criterion(out_Y, out_L, inputs,inputimg,labels,labelimg)
        loss = loss_P+loss_F+loss_V+loss_T+loss_LP+loss_LV
        loss_his_P.append(loss_P.item())
        loss_his_F.append(loss_F.item())
        # loss_his_V.append(loss_V.item())
        loss_his_T.append(loss_T.item())
        loss_his_LP.append(loss_LP.item())
        # loss_his_LV.append(loss_LV.item())


        # 反向传播 参数优化
        loss.backward()

        # for name,param in CMNet.named_parameters():   # 打印梯度信息
        #         print('层:',name,param.size())
        #         print('权值梯度',param.grad)

        train_optimizer.step()


        # 预测训练时间
        tt=time.time() - since
        nowbatchnum = epoch*allbatch+batchid+1
        pre_batch_time = tt/nowbatchnum
        time_oneepoch = pre_batch_time*allbatch
        time_last = pre_batch_time*(allbatch-batchid-1)+time_oneepoch*(epochs-epoch-1)
        print("剩余训练时间：{:.0f}h {:.0f}m {:.0f}s   一轮:{:.0f}h {:.0f}m {:.0f}s".format( time_last // 3600, (time_last%3600) // 60, time_last % 60,
        time_oneepoch // 3600, (time_oneepoch%3600) // 60, time_oneepoch % 60))

        batchid = batchid + 1

        # 每10批输出一次扭曲矩阵
        if batchid % 10 == 0:
            np.set_printoptions(precision=3)
            print(torch.squeeze(out_Y[0,0,:]).cpu().detach().numpy())
        print('*' * 20)

    # 学习策略,更新学习率
    train_scheduler.step()

    if (epoch+1) % 1 == 0:
        # 画损失函数图
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)
        # ax5 = plt.subplot(2,3,6)
        plt.sca(ax1)
        plt.plot(loss_his_P[epoch*allbatch:])  
        plt.xlabel("batch")
        plt.ylabel("lossP")
        plt.sca(ax2)
        plt.plot(loss_his_F[epoch*allbatch:])
        plt.xlabel("batch")
        plt.ylabel("lossF")
        # plt.sca(ax3)
        # plt.plot(loss_his_V[epoch*allbatch:])
        # plt.xlabel("batch")
        # plt.ylabel("lossV")
        plt.sca(ax3)
        plt.plot(loss_his_T[epoch*allbatch:])
        plt.xlabel("batch")
        plt.ylabel("lossT")
        plt.sca(ax4)
        plt.plot(loss_his_LP[epoch*allbatch:])
        plt.xlabel("batch")
        plt.ylabel("lossLP")
        # plt.sca(ax5)
        # plt.plot(loss_his_LV[epoch*allbatch:])
        # plt.xlabel("batch")
        # plt.ylabel("lossLV")
        plt.savefig('img/lossline_'+str(epoch+1)+'.png')
        plt.clf()  #清除图像


    if (epoch+1) % 5 == 0:
        # 模型名称‘序列长度_数据集_参与损失_轮数’
        torch.save(CMNet.state_dict(), 'model/40_minidata_lossPF0TL1_'+str(epoch+1)+'.pth')
        print('网络参数已存储')


# 画损失函数图
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)
# ax5 = plt.subplot(2,3,6)
plt.sca(ax1)
plt.plot(loss_his_P[epoch*allbatch:])  
plt.xlabel("batch")
plt.ylabel("lossP")
plt.sca(ax2)
plt.plot(loss_his_F[epoch*allbatch:])
plt.xlabel("batch")
plt.ylabel("lossF")
# plt.sca(ax3)
# plt.plot(loss_his_V[epoch*allbatch:])
# plt.xlabel("batch")
# plt.ylabel("lossV")
plt.sca(ax3)
plt.plot(loss_his_T[epoch*allbatch:])
plt.xlabel("batch")
plt.ylabel("lossT")
plt.sca(ax4)
plt.plot(loss_his_LP[epoch*allbatch:])
plt.xlabel("batch")
plt.ylabel("lossLP")
# plt.sca(ax5)
# plt.plot(loss_his_LV[epoch*allbatch:])
# plt.xlabel("batch")
# plt.ylabel("lossLV")
plt.savefig('img/lossline.png')
plt.clf()  #清除图像


    