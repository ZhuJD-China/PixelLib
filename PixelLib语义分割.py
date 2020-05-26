# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : ZhuJD
# @FILE     : PixelLib语义分割.py
# @Time     : 2020/5/25 20:42
# @Software : PyCharm

"""
语义分割：就是把图像中每个像素赋予一个类别标签，用不同的颜色来表示。
实例分割：它不需要对每个像素进行标记，它只需要找到感兴趣物体的边缘轮廓就行。

应用:
无人驾驶汽车视觉系统，可以有效的理解道路场景
医疗图像分割，可以帮助医生进行诊断测试
卫星图像分析

PixelLib实现语义分割
PixelLib在执行语义分割任务时，采用的是Deeplabv3+框架，以及在pascalvoc上预训练的Xception模型。
"""
# 用在pascalvoc上预训练的Xception模型执行语义分割
import time
import pixellib
from pixellib.semantic import semantic_segmentation

# 用于执行语义分割的类，是从pixellib导入的，创建了一个类的实例。
segment_image = semantic_segmentation()

# 调用函数来加载在pascal voc上训练的xception模型(xception模型可以从文末传送门链接处下载)。
segment_image.load_pascalvoc_model("./Model/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

# 这是对图像进行分割的代码行，这个函数包含了两个参数：
# path_to_image：图像被分割的路径。
# path_to_output_image：保存输出图像的路径，图像将被保存在你当前的工作目录中。
# segment_image.segmentAsPascalvoc("path_to_image", output_image_name="path_to_output_image")
start = time.time()
segment_image.segmentAsPascalvoc("./FFli.png", output_image_name="deeplabv3_xception-output.png", overlay=True)
end = time.time()
print(f"Inference Time: {end - start:.2f}seconds")
