import torch
import clip
from PIL import Image
from tqdm import tqdm 
from torch import nn 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
  
outputs = []
with open("/mnt/dl-storage/dg-cephfs-0/public/huangyangke/mocov3/images_path.txt",'r',encoding = 'utf-8') as f:
    images_path = f.readlines()[:5000]
    for image in tqdm(images_path):
        image = image.strip()
        image = preprocess(Image.open(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.encode_image(image)
            output = nn.functional.normalize(output, dim=1)
            outputs.append(output)
torch.save(torch.cat(outputs, dim = 0).cpu(), 'embedding_clip.tensor')
