#### mocov3_infer.py
通过mocov3进行特征提取，并进行相似度判断，预训练模型路径：https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md<br/>
其中数据增强采用优化后的letterbox，较长边放缩到224，并保持长宽比不变，不进行padding，主要目的是为了保证原图信息不缺失也不额外增加干扰（改用resize，可以比较下两个效果）<br/>
尝试加上project，发现效果很差，resnet直接使用2048维度特征效果反而更好<br/>

#### phash_infer.py
通过phash算法进行特征提取，原理refer：https://www.cnblogs.com/ERKE/p/14110372.html<br/>
其中相似度判断通过简单的元素比对进行判断<br/>
算法实现from：https://github.com/idealo/imagededup<br/>
优化点：可以通过其他距离判断方法<br/>

#### clip_infer.py
通过clip的视觉模型抽取图像特征<br/>
依赖于clip库，https://github.com/openai/CLIP，使用需要下载<br/>
