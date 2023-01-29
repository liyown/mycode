import torch.nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])]
                                         )
train_dataset = torchvision.datasets.CIFAR10("../data", train=True, transform=transform, download=True)
eval_dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=transform, download=True)
train_dataloard = DataLoader(train_dataset, batch_size=1024, shuffle=True)
eval_dataloard = DataLoader(eval_dataset, batch_size=1024, shuffle=True)

resnet = torchvision.models.resnet34().cuda()

loss_fn = torch.nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(resnet.parameters(), 1e-3, (0.99, 0.999), eps=1e-8)

write = SummaryWriter("../logs_resnet")
train_step = 0
eval_step = 0

for epoch in range(100):
    train_accuracy = 0
    train_loss = 0
    print("-----------epoch {}-----------".format(epoch))
    resnet.train()
    for data in train_dataloard:
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
