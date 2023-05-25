import torch
import torchvision
import torchvision.utils as vutils
from model import Generator

batch_size = 64
latent_dim = 100
generator = Generator(latent_dim, 3136)
generator.load_state_dict(torch.load("./result/model_params.pth"))

# 生成一批图像
z = torch.randn(batch_size, latent_dim)
fake_images = generator(z)

# 将一维向量还原为图像
fake_images = (fake_images + 1) / 2  # 将像素范围调整为[0, 1]

# 保存图像到本地文件
torchvision.utils.save_image(fake_images, './result/generated_images.png',normalize=True)