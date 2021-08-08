import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from videoResize import getVideoPath_toResize       # 获取视频路径类的函数
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self, vid_fapath, vid_format, step_time, transform=None):
        super(MyDataset,self).__init__()

        self.vid_fapath = vid_fapath        # 父目录
        self.vid_format = vid_format        # 视频格式
        self.step_time = step_time          # 序列串长度
        self.videoPathList = getVideoPath_toResize(self.vid_fapath,self.vid_format) # 获取视频路径类列表，前100为不稳定，后100为稳定，内部按照视频类型顺序排列
        self.transform = transform
        # 每个视频的帧数
        # self.framenumlist = [900, 450, 600, 540, 450, 450, 360, 600, 450, 330, 600, 300, 430, 420, 420, 430, 570, 530, 460, 680, 400, 310, 910, 370, 650, 690, 560, 345, 719, 460, 420, 560, 469, 510, 360, 500, 430, 410, 400, 500, 360, 540, 530, 250, 530, 290, 880, 690, 290, 540, 290, 840, 900, 500, 589, 420, 420, 450, 400, 900, 660, 390, 430, 790, 820, 450, 460, 360, 439, 310, 350, 820, 980, 360, 440, 310, 680, 950, 469, 350, 500, 770, 630, 670, 600, 490, 390, 470, 390, 450, 635, 350, 799, 490, 270, 650, 740, 930, 320, 490]
        self.framenumlist = [900, 450, 600, 540, 450, 450, 360, 600, 450, 330, 600, 300, 430, 420, 420, 430, 570, 530, 460, 680, 400, 310, 910, 370, 650, 690, 560, 345, 719, 460]
        # self.framenumlist = [900]
        self.vidnumlist = []    # 每个视频及其以前的视频总共包含多少个序列串
        sumlen = 0
        for item in self.framenumlist:
            sumlen = sumlen+int(item/(self.step_time-4))    # 因为一次需要取出 self.step_time-4 张图片
            self.vidnumlist.append(sumlen)
        self.sumlen = sumlen                # 所有视频的序列串总数


    def __len__(self):
        return self.sumlen

    def __getitem__(self, idx): # idx为序列串索引
        # 找出目前序列串所在的视频在视频类列表中的下标
        now_vid_idx = 0 # 不稳定视频在列表中的下标，稳定真值视频下标为now_vid_idx+int(len(self.videoPathList)/2)
        for vid_idx in range(len(self.vidnumlist)):
            if self.vidnumlist[vid_idx] > idx:
                now_vid_idx = vid_idx
                break
        # 计算序列串在当前视频里的帧索引
        if now_vid_idx != 0:
            idx_invid = idx - self.vidnumlist[now_vid_idx-1]    # 当前视频里的串索引
        else:
            idx_invid = idx
        frame_invid_idx = idx_invid*(self.step_time-4)          # 在视频中的第一帧索引

        # 格式为(S,C,H,W)
        inputdata = torch.zeros(size=(self.step_time,3,256,256))    # 输入网络的图片数据，经过transform处理
        input_img = torch.zeros(size=(self.step_time,3,256,256))    # 原始不稳定图片数据，用于后面求损失函数
        labeldata = torch.zeros(size=(self.step_time,3,256,256))    # 经过transform处理的地面真值数据
        label_img = torch.zeros(size=(self.step_time,3,256,256))    # 对应的稳定图片数据，用于后面求损失函数

        # 读视频，获取不稳定视频帧
        cap = cv2.VideoCapture(self.videoPathList[now_vid_idx].getAllPath())
        # print(self.videoPathList[now_vid_idx].getAllPath())
        frame_count = 0
        while(True):
            ret, frame = cap.read()
            if ret is False:
                break
            ni = frame_count - frame_invid_idx
            # 如果是需要取出的帧
            if ni >= 0 and ni <= self.step_time-5:
                # 把原始的opencv格式改成tensor(C,H,W)
                orgframe = torch.Tensor(frame.transpose(2,0,1))
                # 图片预处理前需要把opencv转化为PIL Image的格式
                dataframe = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                # 图片预处理
                if self.transform:
                    dataframe = self.transform(dataframe)
                # 第一帧需要复制4次
                if ni == 0:
                    inputdata[0,:]=dataframe
                    inputdata[1,:]=dataframe
                    inputdata[2,:]=dataframe
                    inputdata[3,:]=dataframe
                    input_img[0,:]=orgframe
                    input_img[1,:]=orgframe
                    input_img[2,:]=orgframe
                    input_img[3,:]=orgframe
                inputdata[ni+4,:]=dataframe
                input_img[ni+4,:]=orgframe
            # 取完了
            elif ni > self.step_time-5:
                break
            frame_count = frame_count + 1

        # 获取对应稳定帧
        cap = cv2.VideoCapture(self.videoPathList[now_vid_idx+int(len(self.videoPathList)/2)].getAllPath())
        # print(self.videoPathList[now_vid_idx+int(len(self.videoPathList)/2)].getAllPath())
        frame_count = 0
        while(True):
            ret, frame = cap.read()
            if ret is False:
                break
            ni = frame_count - frame_invid_idx
            if ni >= 0 and ni <= self.step_time-5:
                # 把原始的opencv格式改成tensor(C,H,W)
                orgframe = torch.Tensor(frame.transpose(2,0,1))
                # 图片预处理前需要把opencv转化为PIL Image的格式
                dataframe = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                # 图片预处理
                if self.transform:
                    dataframe = self.transform(dataframe)
                if ni == 0:             # 第一帧需要复制4次
                    labeldata[0,:]=dataframe
                    labeldata[1,:]=dataframe
                    labeldata[2,:]=dataframe
                    labeldata[3,:]=dataframe
                    label_img[0,:]=orgframe
                    label_img[1,:]=orgframe
                    label_img[2,:]=orgframe
                    label_img[3,:]=orgframe
                labeldata[ni+4,:]=dataframe
                label_img[ni+4,:]=orgframe
            elif ni > self.step_time-5:             # 取完了
                break
            frame_count = frame_count + 1
        
        # inputdata和labeldata为预处理后的图片数据，input_img和label_img是不稳定-稳定帧的tensor(C,H,W)BGR格式
        return inputdata,input_img,labeldata,label_img


# test code
if __name__ == '__main__':
    fatherPath = u'/Users/xhpzww/Documents/我的稳定数据集/已处理/256*256'   # 父目录
    videoFormat = '.avi'                                                # 视频格式
    # 数据集的预处理设置
    mytransforms = transforms.Compose([
            transforms.ToTensor(),      # 将PIL.Image转化为tensor，即归一化过程,并图片格式为(C,H,W)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])
    test=MyDataset(fatherPath,videoFormat,20,transform=mytransforms)
    print(len(test))
    x, y, z = test[5]
    print(x.shape)
    print(y.shape)
    print(z.shape)
    dataloder=DataLoader(test, batch_size=8, shuffle=True, num_workers=1)
    for inputs, inputimg, labelimg in dataloder:  # 拿批数据
        inputs = inputs.transpose(0,1)
        inputimg = inputimg.transpose(0,1)
        labelimg = labelimg.transpose(0,1)
        print(inputs.shape)
        print(inputimg.shape)
        print(labelimg.shape)
        break









    # # 计算训练集所有视频的帧数
    # # fatherPath = u'/Users/xhpzww/Documents/我的稳定数据集/已处理/256*256'
    # # videoFormat = '.avi'
    # # videoPathList = getVideoPath_toResize(fatherPath,videoFormat)   # 获取视频路径
    # framenumlist = [900, 450, 600, 540, 450, 450, 360, 600, 450, 330, 600, 300, 430, 420, 420, 430, 570, 530, 460, 680, 400, 310, 910, 370, 650, 690, 560, 345, 719, 460, 420, 560, 469, 510, 360, 500, 430, 410, 400, 500, 360, 540, 530, 250, 530, 290, 880, 690, 290, 540, 290, 840, 900, 500, 589, 420, 420, 450, 400, 900, 660, 390, 430, 790, 820, 450, 460, 360, 439, 310, 350, 820, 980, 360, 440, 310, 680, 950, 469, 350, 500, 770, 630, 670, 600, 490, 390, 470, 390, 450, 635, 350, 799, 490, 270, 650, 740, 930, 320, 490]
    # # for vid in videoPathList:   # 遍历每个路径类
    # #     nowPath = vid.getAllPath()  # 视频路径
    # #     # 获取视频
    # #     cap = cv2.VideoCapture(nowPath)
    # #     frame_count = 0
    # #     while(True):
    # #         ret, _ = cap.read()
    # #         if ret is False:
    # #             break
    # #         frame_count = frame_count + 1
    # #     framenumlist.append(frame_count)
    # print(framenumlist)
    # print(len(framenumlist))
    # # 检测稳定-不稳定视频帧数量是否一致
    # for idx in range(len(framenumlist)):
    #     if idx >= 100:
    #         break
    #     else:
    #         if framenumlist[idx] != framenumlist[idx+100]:
    #             print(idx)
    #             print(framenumlist[idx])
    #             print(framenumlist[idx+100])
    # print(framenumlist[:100])
    # print(len(framenumlist[:100]))
