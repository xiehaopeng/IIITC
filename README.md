# IIITCNET: Intra- and Inter-frame Iterative Temporal Convolutional Networks for Video Stabilization
Official project page for paper, IIITCNET: Intra- and Inter-frame Iterative Temporal Convolutional Networks for Video Stabilization.

we show an example videos to compare our work with other methods.
 ![result demo](./docs/result_demo.gif)

The video can be obtained [here](https://www.youtube.com/watch?v=RiOOc_clpjE) .


## Dataset
 ![dataset demo](./docs/dataset_demo.gif)

You can download our video stabilization dataset from [Google Drive](https://drive.google.com/drive/folders/1PKH6rn8U_I0EZbJdmjzStuMX1Ica4pWS?usp=sharing) 

We provide a real-world video dataset of the pre-print paper 'IIITCNET: Intra- and Inter-frame Iterative Temporal Convolutional Networks for Video Stabilization'.

There are 140 pairs of synchronized videos in this dataset. Each pair of videos includes a jitter video and a stabilized video after hardware stabilization. The length of each video is about 15-30 seconds and frames per second is 30. The resolution is 1920x1080.

According to the motion type, jitter degree and challenging content, the dataset was divided into the following 9 categories:
1. simple. It only contains the linear motion of the lens, and the scene content is simple.
2. running. It has the violent up and down motion while shaking.
3. quick rotation. It contains blurring and distortion caused by the quick rotation of the lens
4. vehicle. The jitter from vehicle motion was added in videos.
5. parallax. It has big difference in the apparent position of an object viewed along two different lines of sight.
6. depth. The depth transformation in the video is discontinuous.
7. occlusion. It includes large-scale occlusion caused by close-up objects or obstacles.
8. crowd. It has a large number of moving objects in different motion states.
9. low-quality. This kind of video will cause feature extraction failures. It was divided into 4 sub-categories:dark, blur, noise and watermark.


## Usage of this repo
### Prerequisites
- Python 3.5
- CUDA 9.0
- pytorch 1.1.0
- torchvision 0.3.0
- cudatoolkit 9.0
- numpy
- ...

### Data preparation
You can download [deepstab](http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip) (7.9GB) or [our dataset](https://drive.google.com/drive/folders/1PKH6rn8U_I0EZbJdmjzStuMX1Ica4pWS?usp=sharing) (10.2GB) as training data.

We use the public [bundled camera path dataset](http://liushuaicheng.org/SIGGRAPH2013/database.html) in our experiment. You can also use your own shake videos to verify the effect of the model.

After downloading, You need to modify the video path of the code in every ".py" file.

### Training
You need to modify the save path of the model, which is in the “./model” folder by default.

```python train_model.py```

### Testing
Before testing, you can train the model yourself or use the model parameters we have trained. [our dataset](https://drive.google.com/drive/folders/1Zt0TvY7f4opXXxzyHsph0sPV9ufH0qkZ?usp=sharing)

```python test_model.py```

## Contact
Please contact the first author Haopeng Xie (xhpzww0822@163.com) for any questions.