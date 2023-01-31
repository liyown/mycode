# 定义模型
import torch
import torchvision
from torch import nn, optim

DATASET_PATH = '../data'
SAVE_PATH = '../model_params'
data_name = "CIFAR100"

model_ft = torchvision.models.resnet18(pretrained=False)

# 修改模型
model_ft.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
model_ft.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
num_ftrs = model_ft.fc[0].in_features  # 获取（fc）层的输入的特征数
model_ft.fc = nn.Linear(num_ftrs, 100, bias=True)

optimizer = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
state = {
    'state_dict': model_ft.state_dict(),
    'best_acc': 0,
}
torch.save(state, "{}/{}_best_cnn_model.pth".format(SAVE_PATH, data_name))

now_state = {
    'state_dict': model_ft.state_dict(),
    'optimizer': optimizer.state_dict(),
    "epoch": 0

}
torch.save(now_state, "{}/{}_now_cnn_model.pth".format(SAVE_PATH, data_name))
