import torch 
import os
import shutil

images_path_list = []
with open("/mnt/dl-storage/dg-cephfs-0/public/huangyangke/mocov3/images_path.txt",'r',encoding = 'utf-8') as f:
    data = f.readlines()[:5000]
    for line in data:
        line = line.strip()
        images_path_list.append(line)
        
threshold = 0.98
embedding = torch.load('embedding.tensor')[:5000]
logits = torch.einsum('nc,mc->nm', [embedding, embedding])
logits = (logits + 1) / 2
for i in range(logits.shape[0]):
    for j in range(i + 1, logits.shape[1]):
        if logits[i][j] > threshold:
            dir_path = f'/mnt/dl-storage/dg-cephfs-0/public/huangyangke/mocov3/mocov3{threshold}/{i}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            if not os.path.exists(dir_path + '/' + os.path.basename(images_path_list[i])):
                shutil.copy(images_path_list[i],dir_path + '/' + os.path.basename(images_path_list[i]))
            shutil.copy(images_path_list[j],dir_path + '/' + os.path.basename(images_path_list[j]))
