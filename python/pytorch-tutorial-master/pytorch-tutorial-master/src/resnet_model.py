# 使用Resnet18原始model
# 用完整的测试集和验证集
import sys
import time
import copy
import numpy as np
import torch
import torchvision.models
from tqdm import tqdm
from torchvision.transforms import transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    Write = SummaryWriter("../logs_train")
    # 这里面的变量都相当于全局变量 ！！

    # GPU计算
    device = torch.device("cuda")

    #  训练总轮数
    total_epochs = 500
    # 每次取出样本数
    batch_size = 512
    # 初始学习率
    Lr = 0.1

    DATASET_PATH = '../data'
    SAVE_PATH = '../model_params'

    data_name = "CIFAR100"
    best_filename = '{}/{}_best_cnn_model'.format(SAVE_PATH,data_name)  # 文件扩展名在保存时添加
    now_filename = '{}/{}_now_cnn_model'.format(SAVE_PATH, data_name)
    # 准备数据
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
            , transforms.Cutout(n_holes=1, length=16)
            , transforms.RandomCrop(32, padding=8)  # 先四周填充0，在吧图像随机裁剪成32*32
            , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # 准备数据 这里将训练集和验证集写到了一个list里 否则后面的训练与验证阶段重复代码太多
    image_datasets = {
        x: CIFAR100(DATASET_PATH, train=True if x == 'train' else False,
                   transform=data_transforms[x], download=True) for x in ['train', 'valid']}

    dataloaders: dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False
        ) for x in ['train', 'valid']
    }

    # 定义模型
    model_ft = torchvision.models.resnet18(pretrained=False)

    # 修改模型
    model_ft.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model_ft.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    num_ftrs = model_ft.fc[0].in_features  # 获取（fc）层的输入的特征数
    model_ft.fc = nn.Linear(num_ftrs, 100, bias=True)
    model_ft.to(device)
    # 创建损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # 定义优化器
    optimizer = optim.SGD(model_ft.parameters(), lr=Lr, momentum=0.9, weight_decay=5e-4)

    # 训练模型
    # 显示要训练的模型
    # print("==============当前模型要训练的层==============")
    # for name, params in model_ft.named_parameters():
    #     if params.requires_grad:
    #         print(name)

    # 训练模型所需参数
    # 用于记录损失值未发生变化batch数
    counter = 0
    # 记录训练次数
    total_step = {
        'train': 0, 'valid': 0
    }

    best_data = torch.load("{}/{}_best_cnn_model.pth".format(SAVE_PATH, data_name))
    now_data = torch.load('{}/{}_now_cnn_model.pth'.format(SAVE_PATH, data_name))
    # 记录开始时间
    since = time.time()
    # 记录当前最小损失值
    valid_loss_min = 0
    # 保存模型文件的尾标
    save_num = 0
    # 保存最优正确率
    best_acc = best_data["best_acc"]
    optimizer.load_state_dict(now_data["optimizer"])
    model_ft.load_state_dict(best_data["state_dict"])
    now_epoch = now_data["epoch"]

    # Lr = now_data["optimizer"]["param_groups"][0]["lr"]
    for params in optimizer.param_groups:
        params['lr'] = 0.001
    for epoch in range(now_epoch + 1, total_epochs):
        # 动态调整学习率
        if counter / 10 == 1:
            counter = 0
            Lr = Lr * 0.5
            for params in optimizer.param_groups:
                params['lr'] = Lr

        # 在每个epoch里重新创建优化器？？？


        print('-------------Epoch {}/{}------------'.format(epoch, total_epochs))
        # print('-' * 10)
        # 训练和验证 每一轮都是先训练train 再验证valid
        for phase in ['train', 'valid']:
            # 调整模型状态
            if phase == 'train':
                model_ft.train()  # 训练
            else:
                model_ft.eval()  # 验证

            # 记录损失值
            running_loss = 0.0
            # 记录正确个数
            running_corrects = 0

            # 一次读取一个batch里面的全部数据
            loop = tqdm(dataloaders[phase], leave=False, file=sys.stdout)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    loss = loss_fn(outputs, labels)

                    # torch.max() 返回的是一个元组 第一个参数是返回的最大值的数值 第二个参数是最大值的序号
                    _, preds = torch.max(outputs, 1)  # 前向传播 这里可以测试 在valid时梯度是否变化

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()  # 反向传播
                        optimizer.step()  # 优化权重
                        # TODO:在SummaryWriter中记录学习率
                        # ....

                # 计算损失值
                running_loss += loss.item() * inputs.size(0)  # loss计算的是平均值，所以要乘上batchsize，计算损失的总和
                running_corrects += (preds == labels).sum()  # 计算预测正确总个数
                # 每个batch加1次
                total_step[phase] += 1
                loop.set_description(f'Epoch [{epoch}/{total_epochs}]')
                loop.set_postfix(loss=loss.item(), acc=((preds == labels).sum() / len(labels)).item())
            # 一轮训练完后计算损失率和正确率
            epoch_loss = running_loss / len(dataloaders[phase].sampler)  # 当前轮的总体平均损失值
            epoch_acc = float(running_corrects) / len(dataloaders[phase].sampler)  # 当前轮的总正确率



            time_elapsed = time.time() - since
            print('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}[{}] Acc: {:.4f}[{}]'.format(phase, epoch_loss, counter, epoch_acc, counter))
            Write.add_scalar(phase + "Loss", epoch_loss, epoch)
            Write.add_scalar(phase + "Acc", epoch_acc, epoch)
            if phase == 'valid':
                # 得到最好那次的模型
                if epoch_acc > best_acc:  # :

                    best_acc = epoch_acc
                    valid_loss_min = epoch_loss

                    # 保存当前模型
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
                    state = {
                        'state_dict': model_ft.state_dict(),
                        'best_acc': best_acc,
                    }
                    # 保存训练结果
                    # save_num = 0 if save_num > 1 else save_num
                    save_name_t = "{}/{}_best_cnn_model.pth".format(SAVE_PATH, data_name)
                    torch.save(state, save_name_t)  # \033[1;31m 字体颜色：红色\033[0m
                    print("已保存最优模型，准确率:\033[1;31m {:.2f}%\033[0m，文件名：{}".format(best_acc * 100, save_name_t))
                    # save_num += 1

                    counter = counter - 2
                    counter = (abs(counter) + counter) / 2
                else:
                    counter += 1
            now_state = {
                'state_dict': model_ft.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }

            torch.save(now_state, "{}/{}_now_cnn_model.pth".format(SAVE_PATH, data_name))  # \033[1;31m 字体颜色：红色\033[0m

        print('当前学习率 : {:.5f}'.format(optimizer.param_groups[0]['lr']))
        print()

    # 训练结束
    time_elapsed = time.time() - since
    print('任务完成！')
    print('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('最高验证集准确率: {:4f}'.format(best_acc))
    # save_num = save_num - 1
    # save_num = save_num if save_num < 0 else 1

    save_name_t = "{}/{}_best_cnn_model".format(SAVE_PATH, data_name)
    print('最优模型保存在：{}'.format(save_name_t))
