import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms

class FeatExtractModel():
    def __init__(self, onnx_path, model_type = 'mocov3', gpu_id = 0):
        self.gpu_id = gpu_id # 使用第几张gpu
        self.model_type = model_type
        self.img_size = 224
        self.init_runtime(onnx_path)
        self.get_augment()
        # 获取模型输入名、输出名
        self.input_name = self.runtime.get_inputs()[0].name
        self.output_name = self.runtime.get_outputs()[0].name   
             
    def init_runtime(self, onnx_path):
        self.runtime = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        
    def get_augment(self):
        if self.model_type == 'mocov3' or self.model_type == 'sscd':
            self.augment = transforms.Compose([
                transforms.Resize([self.img_size,self.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        elif self.model_type == 'isc':
            self.augment = transforms.Compose([
                transforms.Resize([self.img_size,self.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5),),
            ])
    
    # 特征归一化
    def feat_norm(self, feats):
        row_norms = np.linalg.norm(feats, axis=1)
        feats_normalized = feats / row_norms[:, np.newaxis]
        return feats_normalized
    
    def __call__(self, img, is_batch = False, l2_norm = True):
        if not is_batch:
            if isinstance(img, str):
                img = Image.open(img)
            else:
                img = img[:,:,::-1]
                img = Image.fromarray(img)
            img = self.augment(img)
            c,h,w = img.shape
            img = img.view(-1, c, h, w)
        else:
            img = [self.augment(Image.fromarray(im[:,:,::-1]))[None] for im in img]
            img = torch.cat(img, dim = 0)
        img = img.numpy()
        feat = self.runtime.run([self.output_name], {self.input_name: img})[0]
        # 是否进行特征归一化
        if l2_norm:
            feat = self.feat_norm(feat)
        return feat

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return res # -1~1

if __name__ == '__main__':
    image1 = cv2.imread('/mnt/cluster/huangyangke/comment/dataset/template.jpg')
    image2 = cv2.imread('/mnt/cluster/huangyangke/comment/dataset/template2.jpg')
    model_path = '/mnt/cluster/huangyangke/comment/moco/moco.onnx' 
    featEncoder = FeatExtractModel(model_path)
    feat1 = featEncoder(image1)
    feat2 = featEncoder(image2)
    sim = np.dot(feat1, np.array(feat2).T)
    print(sim)
