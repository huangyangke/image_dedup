import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models

state_dict = torch.load('/home/test/huangyangke/code/mocov3/r-50-1000ep.pth.tar')['state_dict']
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
x = torch.empty(1,3,224,224)
model.eval()
with torch.no_grad():
    print(model(x).shape)
