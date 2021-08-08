"""用来批量处理视频文件，缩放视频帧到目标尺寸大小，获得训练数据"""
import cv2
import numpy as np


class videoPath(object):
    """稳定视频路径类"""
    def __init__(self, fatherPath, videoType, stabType,videoNum,videoFormat):
        self.fatherPath = fatherPath    # 父路径 str
        self.videoType = videoType      # 视频类别 str
        self.stabType = stabType        # 稳定类别 str
        self.videoNum = videoNum        # 视频编号 int
        self.videoFormat = videoFormat  # 视频格式 str

    def getAllPath(self):
        return self.fatherPath+'/'+self.videoType+'/'+self.stabType+str(self.videoNum)+self.videoFormat

    def changeFatherPath(self,target):
        self.fatherPath = target

    def changeVideoFormat(self,target):
        self.videoFormat = target
        

def getVideoPath_toResize(fatherPath,videoFormat):
    """输入父目录和文件格式，获取视频文件的路径类"""
    videoTypeList = [u'简单',u'跑步',u'快速旋转',u'交通工具',u'大视差',u'不连续深度',u'近距离遮挡',u'人群']
    # videoTypeList = [u'简单']
    stabTypeList = [u'不稳定',u'稳定']
    resizePathList = [] # 存放所有视频地址
    for stabType in stabTypeList:
        for videoType in videoTypeList:
            # 确定每类视频数量
            videoNum = 30 if videoType == u'简单' else 10
            # videoNum = 1 if videoType == u'简单' else 10
            for num in range(videoNum):
                vid = videoPath(fatherPath,videoType,stabType,num+1,videoFormat)
                resizePathList.append(vid)


    return resizePathList


def resizeVideo(videoPathList,targetShape,targetFatherPath):
    """根据原文件路径类、目标尺寸大小和目标父路径，对视频帧进行缩放并存储"""
    for vid in videoPathList:   # 遍历每个路径类
        nowPath = vid.getAllPath()  # 原视频地址
        vid.changeFatherPath(targetFatherPath)
        vid.changeVideoFormat('.avi')
        targetPath = vid.getAllPath()
        print(nowPath)
        print(targetPath)

        # 获取视频
        cap = cv2.VideoCapture(nowPath)
        # 创建VideoWriter，这里要确保目标路径的文件夹存在才能写入
        videowriter = cv2.VideoWriter(targetPath, cv2.VideoWriter_fourcc(*'MJPG'), 30, targetShape)

        success, _ = cap.read()

        while success:
            success, img1 = cap.read()
            try:
                # 缩放
                img = cv2.resize(img1, targetShape, interpolation=cv2.INTER_LINEAR)
                videowriter.write(img)
            except:
                break
        videowriter.release()


# test code
if __name__ == '__main__':
    # 定义父目录和文件格式
    fatherPath = u'/Users/xhpzww/Documents/我的稳定数据集/已处理/1920*1080'
    videoFormat = '.mp4'
    # 获取视频路径
    videoPathList = getVideoPath_toResize(fatherPath,videoFormat)
    targetShape = (256,256) # 目标尺寸
    targetFatherPath = u'/Users/xhpzww/Documents/我的稳定数据集/已处理/256*256'
    # 缩放并存储
    resizeVideo(videoPathList,targetShape,targetFatherPath)