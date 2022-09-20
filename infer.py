import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import cv2
import numpy as np
from PIL import Image

def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

#scaleup=True 表示可以向上scale
#auto=True 表示padding到可以整除32即可
#auto=False scaleFill=False表示padding到new_shape大小
#auto=False scaleFill=True表示直接原图缩放到new_shape大小
def letterbox(im, new_shape=(224, 224), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
    dw = 0
    dh = 0

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

state_dict = torch.load('/mnt/dl-storage/dg-cephfs-0/public/huangyangke/mocov3/r-50-1000ep.pth.tar')['state_dict']
linear_keyword = 'fc'
for k in list(state_dict.keys()):
    # retain only base_encoder up to before the embedding layer
    if k.startswith('module.base_encoder'):
        # remove prefix
        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]
    
model = torchvision_models.__dict__['resnet50']()
model.fc = _build_mlp(2, 2048, 4096, 256)
model.load_state_dict(state_dict, strict=True)
# x = torch.empty(1,3,224,224)
# model.eval()
# with torch.no_grad():
#     print(model(x).shape)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

augmentation = transforms.Compose([
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

image1 = cv2.imread('/mnt/dl-storage/dg-cephfs-0/openai/cv-team/luxiangzhe/lofter/img_dupicate/cnn_target_imgs_5/104/YkhpaWFOalhaWEtmV29aZG1laitjUTJQZUFPZlcwMDRjNVRHdXFsVkU1ZXA5aUxGeGZXdUR3PT0.jpg')[:,:,::-1]
image1 = letterbox(image1,new_shape=(224, 224))[0]
image1 = Image.fromarray(image1)
image1 = augmentation(image1).unsqueeze(0)

image2 = cv2.imread('/mnt/dl-storage/dg-cephfs-0/openai/cv-team/luxiangzhe/lofter/img_dupicate/cnn_target_imgs_5/104/YkhpaWFOalhaWEtmV29aZG1laitjV1hiTWVnWGFlVEhWeTlDK3E2d3Y0d1BJSDZoTFNsZ3JnPT0.jpg')[:,:,::-1]
image2 = letterbox(image2,new_shape=(224, 224))[0]
image2 = Image.fromarray(image2)
image2 = augmentation(image2).unsqueeze(0)

model.eval()
with torch.no_grad():
    #print(torch.sum(image1 - image2))
    output1 = model(image1)
    output2 = model(image2)
    q = nn.functional.normalize(output1, dim=1)
    k = nn.functional.normalize(output2, dim=1)
    #print(torch.sum(q - k))
    logits = torch.einsum('nc,mc->nm', [q, k])[0]
    print(logits)
