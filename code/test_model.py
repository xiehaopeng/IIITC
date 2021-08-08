"""
测试模型
"""
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms  # 预处理
from IIITCNET import CMStabNet     # CMStabNet循环网络
from Homography import changeFweight


# 数据集的预处理设置
test_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),      # 将PIL.Image转化为tensor，即归一化过程,并图片格式为(C,H,W)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 组织设备
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def getWarpimgs(lenth,inputdata,orgimg,CMNet):
    '''
    把长度为lenth的视频帧序列送入网络中，预测单应变换矩阵，
    '''
    # 准备输入数据
    inputs = torch.stack(inputdata,dim=0)   # (lenth,C,256,256)
    inputs = torch.unsqueeze(inputs,dim=1)   # (lenth,1,C,256,256)
    inputs.to(dev)
    CMNet.eval()
    # 进入网络，获得所有扭曲矩阵参数
    out_Y,out_L = CMNet(inputs) # out_Y输出扭曲参数(lenth,1,6,1,1)和out_L解码后的未剪裁稳定帧(lenth,1,3,256,256)
    changeFweight(out_Y)
    # 透视部分
    part = torch.zeros([lenth,1,3,1,1])
    part[:,:,2,...] = 1
    out_Y = out_Y.cpu()
    # 扭曲变换矩阵
    homos = torch.cat((out_Y, part),2).view(lenth*1,3,3).detach().numpy()
    # ls = []
    # for i in range(lenth):
    #     ls.append(np.identity(3))
    # homos = np.array(ls)
    # 扭曲第i帧
    for i in range(4,len(orgimg)):
        nownum = i+frame_count-lenth    #目前扭曲帧在视频中的帧数
        print('正在扭曲：'+str(vnum)+'  '+str(nownum))
        homo = homos[i] 

        img = orgimg[i]
        # 由于预测的homo是由经过Resize的图像得到的，需要再加上一个尺度变换
        orgsize = img.shape # (H,W,C)
        # 尺度变换矩阵
        R = np.array([[256/orgsize[1], 0, 0], [0, 256/orgsize[0], 0],[0,0,1]])
        homo = np.linalg.inv(R).dot(homo).dot(R)    # H' = R*H*R.I
        # 把所有图片向右下角平移，并且将画布扩大0.5倍，避免扭曲后超出图像范围
        homo[0,2] += orgsize[1]*0.5/2
        homo[1,2] += orgsize[0]*0.5/2
        Homography_I = cv2.warpPerspective(img,homo,(int(orgsize[1]*1.5),int(orgsize[0]*1.5)))
        # !!!!!!保存稳定后的画面!!!!!!
        cv2.imwrite('../all_video_data/CMStab/Regular/'+str(vnum)+'/'+str(nownum).zfill(4)+'.png', Homography_I)


# !!!!!!确定视频编号范围!!!!!!
startnum = 0
endnum = 0
# !!!!!!获取测试视频路径列表!!!!!!
fatherPath = '../all_video_data/orgvideo/Regular_stable/'
videoType = '.mp4'
# fatherPath = '../testdata/mydata/'
# videoType = '.mp4'
videoPathList = []
for videoNum in range(startnum,endnum+1):
    videoPathList.append(fatherPath+str(videoNum)+videoType)
print(videoPathList)

# !!!!!!设置序列长度（按照内存大小来设置，推荐80）!!!!!!
step_time = 80


# 测试开始
vnum = startnum
for videoPath in videoPathList:
    time_start=time.time()
    # !!!!!!加载网络!!!!!!
    CMNet = CMStabNet((256,256), 3, step_time, 1)
    CMNet.load_state_dict(torch.load('model/20_minidata_lossPFVL2_12.pth', map_location='cpu'))
    CMNet = CMNet.to(dev)
    # 保存待处理帧数据
    inputdata=[]
    orgimg=[]

    print('正在处理视频:'+videoPath)
    frame_count = 0 # 记录帧数
    # 读视频
    cap = cv2.VideoCapture(videoPath)
    while(True):
        ret, frame = cap.read()
        if ret is False:
            break
        # 分别保存原始帧图片和输入网络中的预处理后图片
        orgframe = frame
        # 图片预处理前需要把opencv转化为PIL Image的格式
        dataframe = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        # 图片预处理
        dataframe = test_transforms(dataframe)
        # 若是第一帧需要复制4次
        if len(orgimg) == 0:
            for ti in range(0,4):
                inputdata.append(dataframe)
                orgimg.append(orgframe)
        inputdata.append(dataframe)
        orgimg.append(orgframe)
        frame_count += 1

        # 存够合适长度的帧
        if len(orgimg) % step_time == 0:
            # 扭曲帧并保存
            getWarpimgs(step_time,inputdata,orgimg,CMNet)
            # 清空缓存
            inputdata=[]
            orgimg=[]

    lastnum = len(orgimg)
    if lastnum != 0:    # 若缓存区还有剩余
        # ！！！！！！加载新的剩余网络！！！！！！！！
        CMNet = CMStabNet((256,256), 3, lastnum, 1)
        CMNet.load_state_dict(torch.load('model/20_minidata_lossPFVL2_12.pth', map_location='cpu'))
        CMNet = CMNet.to(dev)
        getWarpimgs(lastnum,inputdata,orgimg,CMNet)
        
    # 打印帧数和处理速度
    print('总帧数：'+str(frame_count))
    time_end=time.time()
    print("FPS:" + str(format(frame_count/(time_end-time_start), '.2f')) ) 

    vnum += 1

    





