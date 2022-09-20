import json
import os
import imagededup
from imagededup.methods import PHash
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
print(imagededup.__file__)

myencoder = PHash()
# Generate encodings for all images in an image directory
path = '/mnt/dl-storage/dg-cephfs-0/openai/cv-team/luxiangzhe/lofter/img_dupicate/new_imgs/'
imgs = os.listdir(path)
import time
s1 = time.time()
encodings = []
for img in tqdm(imgs[:10]):
    #得到phash编码值 16进制字符串 总共16个值
    encoding = myencoder.encode_image(image_file=os.path.join(path, img))
    #print(len(encoding),encoding)
    encodings.append(encoding)

#s2 = time.time()
#print((s2 - s1)/1000)
#with open('./1000_test.txt', 'w')as f:
#    for en in encodings:
#        f.write(bin(int(en,16))[2:] + '\n')
#f.close()


for i in range(len(encodings)):
    #转换成01二进制
    en = bin(int(encodings[i],16))
    for j in range(i+1, len(encodings)):
        en2=bin(int(encodings[j],16))
        cnt = 0
        for k in range(2, len(en)):
            #简单的判断下有几个字符串相等
            if en[k] == en2[k]:
                cnt += 1
        print(cnt / 64)
