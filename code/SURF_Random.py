"""pip3 uninstall opencv-python
   pip3 install opencv-contrib-python
"""
import cv2
import numpy as np


def get_good_match(des1,des2):
    '''用K近邻算法匹配特征点'''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    if len(matches) > 4:
        for pair in matches:
            try:
                m, n = pair
            except (ValueError):
                break
            if m.distance < 0.75 * n.distance:
                good.append(m)
    return good


def surf_kp(image):
    '''SURF特征点检测'''
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # print(gray_image.dtype)
    surf = cv2.xfeatures2d_SURF.create(6000)
    kp, des = surf.detectAndCompute(gray_image, None)
    # 返回关键点和描述子
    return kp,des


def SURF_RAN_Match(Ix, Iy):
    """计算两个图片之间的单应变换
       用SURF提取关键点进行匹配，再用RANDOM单应计算匹配来提高精度
       input: 两个需要匹配的图像Ix，Iy格式为(H,W,C)
       output: 返回两个列表pAlist,pBlist来分别表示匹配成功的特征点。
    """
    kp1,des1 = surf_kp(Ix)
    kp2,des2 = surf_kp(Iy)
    goodMatch = []
    if des1 is not None and des2 is not None:
        goodMatch = get_good_match(des1,des2)
    pAlist = []
    pBlist = []
    H = np.eye(3, dtype=float)
    # 如果有足够的特征匹配点，返回内点pA和pB列表;匹配失败则返回空列表
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        # 返回的第一个参数单应H，第二个参数是掩码
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
        # 根据掩码信息，将内点选出来
        for ids in range(status.shape[0]):
            if status[ids] == 1:
                pAlist.append(ptsA[ids].reshape(2))
                pBlist.append(ptsB[ids].reshape(2))
        # 如果匹配特征点数太多则选择前12个
        if len(pAlist) > 20:
            pAlist = pAlist[:20]
            pBlist = pBlist[:20]

    return H,pAlist,pBlist


# test code
if __name__ == '__main__':
    img1 = cv2.imread("1.jpg")
    img2 = cv2.imread("2.jpg")
    _,pA,pB = SURF_RAN_Match(img1,img2)
    # 把列表转化为数组
    pA = np.array(pA)
    pB = np.array(pB)
    # print(pA.shape)
    # print(pB.shape)
    # print(pA)
    # print(pB)
