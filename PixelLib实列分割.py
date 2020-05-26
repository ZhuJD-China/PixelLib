# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : ZhuJD
# @FILE     : PixelLib实列分割.py
# @Time     : 2020/5/25 21:12
# @Software : PyCharm

# PixelLib实现实例分割
# 实例分割——同一类别的对象被赋予不同的colormap
# PixelLib在执行实例分割时，基于的框架是Mask RCNN
from pixellib.instance import instance_segmentation
import time

# # 用于执行实例分割的类，创建了该类的一个实例。
segment_image = instance_segmentation()

# 这是加载 Mask RCNN 模型来执行实例分割的代码
segment_image.load_model("./Model/mask_rcnn_coco.h5")
start = time.time()
# 用边界框(bounding box)来实现分割就可以得到一个包含分割蒙版和边界框的保存图像
segment_image.segmentImage("./FFli.png", output_image_name="mask_rcnn-output.png", show_bboxes=True)
end = time.time()
print(f"Inference Time: {end - start:.2f}seconds")
