import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import cv2
import numpy as np
from PIL import Image
from onnxsim import simplify
import onnx

def init_mocov3(model_path = 'weights/r-50-1000ep.pth.tar'):
    #加载预训练模型权重 这里去掉动量模型
    state_dict = torch.load(model_path)['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    #模型初始化
    model = torchvision_models.__dict__['resnet50']()
    model.fc = nn.Identity()
    model.load_state_dict(state_dict, strict=False)
    return model.eval()

def model2onnx(model, onnx_save_path):
    image = torch.randn([1,3,224,224])
    torch.onnx.export(
        model, (image, ), onnx_save_path,
        opset_version=11, 
        input_names=["input"],
        output_names=["output"],
        # batch动态维度
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    )
    # 简化onnx模型
    model = onnx.load(onnx_save_path)  
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_save_path)
    
def get_augment(img_size = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    augment = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        normalize
    ]) 
    return augment

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return res
    
if __name__ == '__main__':
    ckpt_dir = '/mnt/cluster/huangyangke/video_dna/weights/'
    ckpt_name = 'r-50-1000ep.pth.tar'
    onnx_name = 'r-50-1000ep.onnx'
    model = init_mocov3(ckpt_dir + ckpt_name)
    
    ## 转onnx
    model2onnx(model, ckpt_dir + onnx_name)
    
    ## 测试pytorch输出
    # augment = get_augment()
    # image_list = ['/mnt/cluster/huangyangke/data/image/12000763/00001.jpg',
    #               '/mnt/cluster/huangyangke/data/image/12000763/00002.jpg',
    #               '/mnt/cluster/huangyangke/data/image/12000763/00003.jpg',
    #               '/mnt/cluster/huangyangke/data/image/12000763/00004.jpg']
    # image_tensor = []
    # for img in image_list:
    #     img = Image.open(img)
    #     img = augment(img)
    #     c,h,w = img.shape
    #     img = img.view(-1, c, h, w)
    #     image_tensor.append(img)
    # image_tensor = torch.cat(image_tensor, dim = 0)
    # with torch.no_grad():
    #     feat_matrix = model(image_tensor).numpy()
    # res = get_cos_similar_matrix(feat_matrix, feat_matrix)
    # print(res)   
