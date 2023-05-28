import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# 加载预训练的 VGG19 网络
from torch.nn import Parameter, MSELoss

vgg = models.vgg19(pretrained=True).cuda().features
for param in vgg.parameters():
    param.requires_grad_(False)

# 定义需要迁移的内容图像和风格图像
content_image = Image.open("3.png")
style_image = Image.open("1.png")

# 定义图像变换和标准化(1080, 1620)
transform = transforms.Compose([
    transforms.Resize(480),
    transforms.ToTensor()
])

# 将图像变换为 PyTorch 张量并添加批次维度
content_tensor = transform(content_image).unsqueeze(0).cuda()
style_tensor = transform(style_image).unsqueeze(0).cuda()

result_tensor = content_tensor.detach().squeeze(0)
result_image = transforms.ToPILImage()(result_tensor)
result_image.save("content.jpg")

result_tensor = style_tensor.detach().squeeze(0)
result_image = transforms.ToPILImage()(result_tensor)
result_image.save("style.jpg")

# 将内容图像和风格图像转换为 PyTorch 张量并添加批次维度
# target_tensor = content_tensor.clone().requires_grad_(True).cuda()
target_tensor = Parameter(torch.randn(content_tensor.size(), requires_grad=True).cuda())


# 定义内容损失函数
def content_loss(base_content, target):
    return torch.mean((base_content - target) ** 2)


# 定义风格损失函数
def gram_matrix(input_tensor):
    batch_size, channels, height, width = input_tensor.size()
    features = input_tensor.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)


def style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return torch.mean((gram_style - gram_target) ** 2)


# 计算内容特征和风格特征
content_layers = ['30']
style_layers = ['0', '5', '10', '19', '28']

content_features = {}
style_features = {}


def get_features(model, tensor, layer_names):
    features = {}
    for name, layer in model._modules.items():
        tensor = layer(tensor)
        if name in layer_names:
            features[name] = tensor
    return features


content_features = get_features(vgg, content_tensor, content_layers)
style_features = get_features(vgg, style_tensor, style_layers)

mse = MSELoss()

# 计算风格图像的 Gram 矩阵
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# 定义优化器和超参数
optimizer = optim.Adam([target_tensor], lr=0.01)
epochs = 50
steps_per_epoch = 200

# 进行风格迁移
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # 计算目标图像的特征
        target_features = get_features(vgg, target_tensor, content_layers + style_layers)

        # # 计算损失函数
        content_loss_value = 0
        for content_layer in content_layers:
            content_loss_value += content_loss(target_features[content_layer], content_features[content_layer])
        content_loss_value *= 1e-1

        style_loss_value = 0
        for style_layer in style_layers:
            target_feature = target_features[style_layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[style_layer]
            style_loss_value += style_loss(target_gram, style_gram)

        content_loss_v = mse(target_tensor, content_tensor)
        total_loss = content_loss_value + style_loss_value * 1e-3 + content_loss_v * 1e-2

        # 计算梯度并更新图像
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # 将像素值限制在 0 和 1 之间
    target_tensor.data.clamp_(0, 1)
    # 将 PyTorch 张量转换为图像并保存结果
    result_tensor = target_tensor.detach().squeeze(0)
    result_image = transforms.ToPILImage()(result_tensor)
    result_image.save("result{}.jpg".format(epoch))

    # 每个 epoch 结束后显示当前的损失值
    print("Epoch: {}, Loss: {}".format(epoch + 1, total_loss))

# 将 PyTorch 张量转换为图像并保存结果
result_tensor = target_tensor.detach().squeeze(0)
result_image = transforms.ToPILImage()(result_tensor)
result_image.save("result.jpg")