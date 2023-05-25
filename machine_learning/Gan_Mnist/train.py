import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator

latent_dim = 100
lr = 0.0001
batch_size = 64
num_epochs = 50

# 加载数据集
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
generator = Generator(latent_dim, 3136).cuda()
discriminator = Discriminator().cuda()
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
# 训练GAN网络
for epoch in range(num_epochs):

    for i, (real_images, _) in enumerate(trainloader):
        real_images = real_images.cuda()
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, latent_dim).cuda()
        # 训练判别器
        real_labels = torch.ones(batch_size, 1).cuda()
        fake_labels = torch.zeros(batch_size, 1).cuda()
        fake_images = generator(z)

        outputs_real = discriminator(real_images)
        outputs_fake = discriminator(fake_images)
        loss_d_real = criterion(outputs_real, real_labels)
        loss_d_fake = criterion(outputs_fake, fake_labels)
        loss_d = loss_d_real + loss_d_fake

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # 训练生成器
        fake_images = generator(z)
        outputs_fake = discriminator(fake_images)
        loss_g = criterion(outputs_fake, real_labels)

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # 打印损失值
        if (i + 1) % 100 == 0:
            # 将一维向量还原为图像
            fake_images = fake_images.view(batch_size, 1, 28, 28)
            fake_images = (fake_images + 1) / 2  # 将像素范围调整为[0, 1]
            # 保存图像到本地文件
            torchvision.utils.save_image(fake_images, './result/generated_images_{0}_{1}.png'.format(epoch + 1, i+1),
                                         normalize=True)
            torch.save(generator.state_dict(), './result/model_params.pth'.format(epoch + 1))
            print('Epoch [{}/{}], Step [{}/{}], Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(trainloader), loss_d.item(), loss_g.item()))
