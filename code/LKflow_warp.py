import cv2
import numpy as np


def compute_TVL1flow(prev, curr, bound=15):
    """ 计算TVL1光流
        opencv4.0以下版本API接口cv2.DualTVL1OpticalFlow_create()
        opencv4.0以上版本API接口使用cv2.optflow.DualTVL1OpticalFlow_create()
        安装opencv时候，不仅要安装opencv-python版本，还需要安装pencv-contrib-python，并且两者安装版本最好一致
    """
    TVL1=cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
 
    # flow = (flow + bound) * (255.0 / (2 * bound))
    # flow = np.round(flow).astype(int)
    # flow[flow >= 255] = 255
    # flow[flow <= 0] = 0
 
    return flow



# test code
if __name__ == '__main__':
    image1 = cv2.imread("img/0015.png")
    image2 = cv2.imread("img/0016.png")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    flow = compute_TVL1flow(image1,image2)

    print(flow)
