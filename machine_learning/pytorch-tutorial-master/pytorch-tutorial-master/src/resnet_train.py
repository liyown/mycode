import torch.nn
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from tqdm import tqdm

transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])]
                                         )
train_dataset = torchvision.datasets.CIFAR10("../data", train=True, transform=transform, download=True)
print(train_dataset.data)
eval_dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=transform, download=True)
train_dataloard = DataLoader(train_dataset, batch_size=256, shuffle=True)
eval_dataloard = DataLoader(eval_dataset, batch_size=256, shuffle=True)

resnet = torchvision.models.resnet18().cuda()
resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
resnet.fc = torch.nn.Sequential(nn.Dropout(0.1),
                                nn.Linear(in_features=512, out_features=10, bias=True))

resnet = resnet.cuda()
print(resnet)
loss_fn = torch.nn.CrossEntropyLoss().cuda()
lr = 0.005
optimizer = torch.optim.SGD(resnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
write = SummaryWriter("../logs_resnet")
train_step = eval_step = pre_epoch = 21
for epoch in range(pre_epoch, 100):
    start_time = time.time()
    train_accuracy = 0
    train_loss = 0
    print("-----------epoch {}-----------".format(epoch))
    resnet.train()
    resnet.load_state_dict(torch.load("../model_params/resnet.pth"))
    for data in tqdm(train_dataloard, leave=False):

        img, label = data
        img, label = img.cuda(), label.cuda()
        output = resnet(img)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (output.argmax(1) == label).sum()
        train_accuracy = train_accuracy + accuracy
        train_loss = train_loss + loss.item()
    # if epoch % 50 == 0:
    #     lr = optimizer.state_dict()['param_groups'][0]['lr'] * (0.1 ** (epoch // 50))
    #     for params in optimizer.param_groups:
    #         params['lr'] = lr
    end_time = time.time()
    print(end_time - start_time)
    torch.save(resnet.state_dict(), "../model_params/resnet.pth")
    print("train_accuracy {}".format(train_accuracy / len(train_dataset)))
    print("train_loss {}".format(train_loss / len(train_dataloard)))
    write.add_scalar("train_loss", train_loss / len(train_dataloard), train_step)
    write.add_scalar("train_accuracy", train_accuracy / len(train_dataset), eval_step)
    train_step = train_step + 1

    resnet.eval()
    eval_loss = 0
    eval_accuracy = 0
    with torch.no_grad():
        for data in eval_dataloard:
            img, targets = data
            if torch.cuda.is_available():
                img = img.cuda()
                targets = targets.cuda()
            outputs = resnet(img)
            loss = loss_fn(outputs, targets)

            accuracy = (outputs.argmax(1) == targets).sum()
            eval_accuracy = eval_accuracy + accuracy
            eval_loss = eval_loss + loss
        print("eval_accuracy {}".format(eval_accuracy / len(eval_dataset)))
        print("eval_loss {}".format(eval_loss / len(eval_dataloard)))
        write.add_scalar("eval_loss", eval_loss / len(eval_dataloard), eval_step)
        write.add_scalar("eval_accuracy", eval_accuracy / len(eval_dataset), eval_step)

        eval_step = eval_step + 1


