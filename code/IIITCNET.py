"""
环境：Python 3.5 + CUDA 9.0 + pytorch 1.1.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
"""

import time
import torch.nn.functional as F
import torch.nn as nn
import torch
from myLoss import Stab_LossFunc    # 损失函数

# 针对Conv层和BatchNorm层进行不同的权重参数初始化方法
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)    # m.weight.data表示需要初始化的权重。nn.init.normal_()表示随机初始化采用正态分布，均值为0，标准差为0.02.
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)                # nn.init.constant_()表示将偏差定义为常量0 


class featureEncoder(nn.Module):
    """特征提取编码器
       由三组卷积下采样层+一组单独卷积组成
    """

    def __init__(self, input_chans):
        """Args:
           input_chans：int类型 输入帧It的通道个数
        """
        super(featureEncoder, self).__init__()

        self.input_chans = input_chans
        self.conv1 = nn.Conv2d(in_channels=self.input_chans, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.rl1 = nn.LeakyReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.rl2 = nn.LeakyReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.rl3 = nn.LeakyReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.rl4 = nn.LeakyReLU()

    def forward(self, input, isresidual = False):
        """ input: 输入帧It(Batch,3,256,256)
            isresidual: bool类型 决定是否输出残差特征

            output: 提取到的特征BIt(Batch,128,32,32)
            residual1, residual2, residual3:传给解码器的三个残差特征
        """
        x1 = input
        residual1 = self.rl1(self.bn1(self.conv1(x1)))
        x2 = self.mp1(residual1)
        residual2 = self.rl2(self.bn2(self.conv2(x2)))
        x3 = self.mp2(residual2)
        residual3 = self.rl3(self.bn3(self.conv3(x3)))
        x4 = self.mp3(residual3)
        output = self.rl4(self.bn4(self.conv4(x4)))
        if isresidual:
            return output, (residual1, residual2, residual3)
        else:
            return output


class memorySelector(nn.Module):
    """状态记忆选择器
    """

    def __init__(self, shape, input_chans, filter_size, num_features):
        """
        input Args:
          shape: int类型的元组，输入特征BIt的尺寸大小 也是ht和ct的高度和宽度
          input_chans：int类型 输入特征BIt的通道个数
          filter_size: int类型 卷积核的大小
          num_features: int类型 状态的通道数（类似于隐藏层大小）
        """
        super(memorySelector, self).__init__()

        self.shape = shape
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2       # 使卷积之后的尺寸与输入保持一直
        # 用一个卷积层的多个卷积核的方式同时完成ConvLSTM的门计算
        self.conv = nn.Conv2d(  in_channels = self.input_chans + self.num_features,
                                out_channels = 4 * self.num_features, 
                                kernel_size = self.filter_size, 
                                stride = 1, 
                                padding = self.padding)

    def forward(self, input, hidden_state):
        """
        input Args:
          input: 输入数据为It经过特征提取得到的BIt 大小如下 Batch,Chans,H,W
          hidden_state: 一个元组，包含上一个时间步的hidden_state=(ht-1,ct-1)

        output Args:
          (next_h,next_c): 为下一个时间步提供的状态元组
        """

        hidden, c = hidden_state                    # hidden and c 是有通道的图片形式
        combined = torch.cat((input, hidden), 1)    # 在通道纬度上进行拼接
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # 返回了四个门数据在通道纬度上的组合，需要依次分解开来
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)

        return (next_h,next_c)

    # 状态层参数初始化（h，c）
    def init_hidden(self, batch_size):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        return (torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1]).to(dev),
                torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1]).to(dev))
                

class parametersRegression(nn.Module):
    """扭曲参数回归器
       由一个平均池和一个1*1卷积核的卷积层组成
    """

    def __init__(self, shape, input_chans):
        """
        input Args:
          shape: 元组，输入状态特征的长宽尺寸
          input_chans：int类型 输入状态特征的通道个数
        """
        super(parametersRegression, self).__init__()

        self.shape = shape
        self.input_chans = input_chans
        self.pool = nn.AvgPool2d(self.shape[0])
        self.conv1x1 = nn.Conv2d(in_channels=self.input_chans , out_channels=6, kernel_size=1, stride=1)


    def forward(self, input):
        """ input: 输入当前隐藏状态(Batch,128,32,32)
            output_Y:  预测的扭曲矩阵参数
        """
        x = self.pool(input)
        output_Y = self.conv1x1(x)
        return output_Y


class frameDecoder(nn.Module):
    """帧解码器
       由三组反卷积+一个1*1卷积组成
    """

    def __init__(self, shape, input_chans):
        """
        input Args:
          shape: 元组，输入状态特征的长宽尺寸
          input_chans：int类型 输入状态特征的通道个数
        """
        super(frameDecoder, self).__init__()

        self.shape = shape
        self.input_chans = input_chans
        self.dconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.rl1 = nn.LeakyReLU()
        self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.rl2 = nn.LeakyReLU()
        self.dconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.rl3 = nn.LeakyReLU()
        # 最后是1*1卷积将通道数转化为3
        self.conv1x1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
        


    def forward(self, input, residual1, residual2, residual3):
        """ input: 输入当前隐藏状态(Batch,128,32,32)
            output_L:  解码未剪裁的稳定帧(Batch,3,256,256)
        """
        x1 = input
        x1 = self.dconv1(x1)
        # 拼接
        temp1 = torch.cat((x1, residual1), dim=1)
        x2 = self.rl1(self.bn1(self.conv1(temp1)))

        x2 = self.dconv2(x2)
        temp2 = torch.cat((x2, residual2), dim=1)
        x3 = self.rl2(self.bn2(self.conv2(temp2)))

        x3 = self.dconv3(x3)
        temp3 = torch.cat((x3, residual3), dim=1)
        x4 = self.rl3(self.bn3(self.conv3(temp3)))

        output_L = self.conv1x1(x4)
        return output_L


class CMStabNet(nn.Module):
    """由
       特征提取编码器
       状态记忆选择器
       扭曲参数回归器
       帧解码器
       组成
    """

    def __init__(self, shape, input_chans, step_time, batch_size):
        """
        input Args:
          shape: int类型的元组，输入帧It尺寸大小
          input_chans：int类型 输入帧It的通道个数
          batch_size: int类型 批数量
          step_time: int类型 单次更新参数循环单元迭代的次数
        """
        super(CMStabNet, self).__init__()

        self.shape = shape
        self.input_chans = input_chans
        self.batch_size = batch_size
        self.step_time = step_time

        # 创建特征编码器
        self.encoder = featureEncoder(input_chans)
        # 创建记忆选择器
        self.selector = memorySelector((32,32), input_chans=128, filter_size=3, num_features=128) 
        # 创建参数回归器
        self.regression = parametersRegression((32,32), 128)
        # 创建稳定解码器
        self.decoder = frameDecoder((32,32), 128)
        # 初始化网络权重以及CLSTM的中间状态
        self.encoder.apply(weights_init)
        self.selector.apply(weights_init)
        self.regression.apply(weights_init)
        self.decoder.apply(weights_init)
        self.hidden_state = self.selector.init_hidden(self.batch_size)



    def forward(self, input):
        """
        input Args:
          input: 输入数据格式如下 Batch,step,Chans,H,W = (step,Batch,1,256,256)

        output Args:
          Yout: 输出扭曲参数 (Batch,1,1,8)
          Lout: 输出像素稳定帧(Batch,3,256,256)

        """
        # 用于存储两个输出信息
        Yout_list = []
        Lout_list = []
        # 初始化的隐藏状态
        last_hidden_state = self.hidden_state

        for itNum in range(self.step_time):  # 每一次帧间迭代
            
            current_input = input[itNum,...]
            # 向特征编码器中输入不稳定帧，输出特征向量和残差特征
            current_feature, residual = self.encoder(current_input, isresidual=True)
            # 第一次通过记忆选择器，提取不稳定帧的目标稳定状态
            current_hidden_state = self.selector(current_feature, last_hidden_state)
            # 回归不稳定帧的扭曲参数
            current_Y = self.regression(current_hidden_state[0])
            # 开始帧内迭代，将隐藏状态和残差输入解码器中，注意残差的对应位置
            current_L = self.decoder(current_hidden_state[0], residual[2], residual[1], residual[0])
            # 将扭曲参数和稳定版本存储下来
            Yout_list.append(current_Y)
            Lout_list.append(current_L)

            # 第二次通过特征编码器和记忆选择器，更新隐藏状态
            current_feature_L = self.encoder(current_L)
            last_hidden_state = self.selector(current_feature_L, last_hidden_state)
        
        # 将输出合并成一个tensor
        Yout = torch.stack(Yout_list, dim=0)
        Lout = torch.stack(Lout_list, dim=0)

        return Yout,Lout




if __name__ == '__main__':

    ###########用例###########
    # 输入参数设置
    shape = (256, 256)  # H,W
    inp_chans = 3
    batch_size = 2      # 8
    step_time = 20      # 20

    # 组织设备
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 输入数据 S,B,C,H,W
    inputdata = torch.rand(step_time, batch_size, inp_chans, shape[0], shape[1])
    inputdata.to(dev)
    # 真值标签
    truth = torch.rand(step_time, batch_size, inp_chans, shape[0], shape[1])
    truth.to(dev)

    # 创建网络
    CMNet = CMStabNet(shape, inp_chans, step_time, batch_size)
    CMNet.to(dev)

    # 前向传播
    print("正在执行前向传播...",end=" ")
    time_start=time.time()
    out_Y,out_L = CMNet(inputdata)
    time_end=time.time()
    print("前向传播总用时:" + str(format(time_end-time_start, '.2f')) + "s") 

    # 计算损失
    print("正在计算损失...")
    time_start=time.time()
    criterion = Stab_LossFunc(step_time, batch_size)
    loss = criterion(out_Y,inputdata,truth)
    time_end=time.time()
    print("计算损失总用时:" + str(format(time_end-time_start, '.2f')) + "s") 

    # 反向传播梯度
    print("正在执行反向传播...",end=" ")
    time_start=time.time()
    loss.backward()
    time_end=time.time()
    print("反向传播总用时:" + str(format(time_end-time_start, '.2f')) + "s") 

    

