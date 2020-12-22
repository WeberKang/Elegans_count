##基于深度神经网络的线虫计数v2.0
#运行环境：
Windows 10 x64/x86
Windows 7   x64/x86

#编译环境：
Windows 10 x64
Python 3.8

#依赖环境：
AVX2指令集CPU
Microsoft Visual C++ 2015-2019 Redistributable（x64/x86）

#实验环境：
ISO100
2448x1920
0.63x and 3.2x

#概述：
基于深度神经网络的线虫计数（TF_Elegans_count）是一个基于TensorFlow和OpenCV开发的一套线虫计数软件，利用OpenCV对线虫图像边缘的提取后基于训练好的的深度神经网络模型来对线虫进行计数。

#背景：
在实验室中很多线虫计数的任务，枯燥乏味，人工计数还容易出错效率低下，利用计算机来提高效率

#文件说明：
TF_Elegans_count/
	TF_Elegans_count/线虫识别.exe	GUI程序主体
	TF_Elegans_count/Model		模型文件
	TF_Elegans_count/Image		测试图片文件
	TF_Elegans_count/LICENSE.txt		使用许可文件

#操作说明：
1.点击exe文件运行
2.读取Model文件
3.加载模型
4.加载储存图像的文件夹
5.点击列表图片名开始运行判断
6.当图片中有多条重叠线虫时，程序会弹出图片和选择框人工判断条数
7.右侧会出现最后的判断结果1条虫为绿色边框2条虫为蓝色3条及以上为黄色蓝色为识别到图像边缘可能存在半条虫的情况
8.检查无误后点击显示判断为没有线虫或者边缘区域弹出识别为0条虫的区域
9.手动对计数加减后在左下角得到结果
10.如果需要继续判断点击列表中其他图片名继续判断

#高阶：
如果您想训练新的模型
https://github.com/WeberKang/Elegans_count中train.py、utils_paths.py、predict.py自行训练

#演示视频：
演示视频v2.0.mp4

#版本更新内容：
v0.1 -基本内容开发实现判断
v0.2 -使用tkinter开发GUI
v1.0 -换用Qt开发GUI
v2.0 -优化部分体验
#维护者：
WeberKang（康伟博）https://github.com/WeberKang Email：kangweber@mail.ynu.edu.cn

#指导老师：
彭城 Email：chengpeng@ynu.edu.cn

#贡献者：
Shiny_X（夏家乐）https://github.com/Skies0

#使用许可：
GNU General Public License v2.0





