'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from models.resnet import ResNet18
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, dropblock_prob=0.15, dropblock_size=3, distillation_weight=0.0):
        self.dropblock_prob = dropblock_prob
        self.dropblock_size = dropblock_size
        self.distillation_weight = distillation_weight
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0
        self.start_epoch = 0
        self.net = None
        self.criterion = None
        self.last_file = None

    def main(self):

        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        self.args = parser.parse_args()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # Data
        # print('==> Preparing data..')
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=128, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        # print('==> Building model..')
        # net = VGG('VGG19')
        self.net = ResNet18(dropblock_prob=self.dropblock_prob, dropblock_size=self.dropblock_size)
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        # net = SimpleDLA()
        self.net = self.net.to(self.device)
        with open('resnet18.txt', 'w') as f:
            f.write(str(self.net))

        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            self.filename = "ckpt_acc83.90_e66_dbs3_dbp0.15_dw0.0.pth"
            self.checkpoint = torch.load(f'./checkpoint/{self.filename}')
            self.net.load_state_dict(self.checkpoint['net'])
            self.best_acc = self.checkpoint['acc']
            self.start_epoch = self.checkpoint['epoch']

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr,
                            momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.run_name = f"dbp{self.dropblock_prob}_dbs{self.dropblock_size}_dw{self.distillation_weight}"
        print(self.run_name)
        wandb.init(project="hinton", 
                mode="run",
                name = self.run_name,
                config={
            "dataset": "cifar",
            "dropblock_prob": self.dropblock_prob,
            "dropblock_size": self.dropblock_size,
            "distillation_weight": self.distillation_weight,
            
        })
        for epoch in range(self.start_epoch, self.start_epoch+200):
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

        with open('results.txt', 'a') as f:
            f.write(f"{self.dropblock_prob}, {self.dropblock_size}, {self.distillation_weight}, {epoch}, {self.best_acc}\n")
        wandb.finish()

    # Training

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(self.trainloader), total=len(self.trainloader))
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if wandb.run.mode != "disabled":
                wandb.log({"train_loss": train_loss/(batch_idx+1), "train_acc": 100.*correct/total})


    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.testloader), total=len(self.testloader))
            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                if wandb.run.mode != "disabled":
                    wandb.log({"test_loss": test_loss/(batch_idx+1), "test_acc": 100.*correct/total})


        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            filename = './checkpoint/ckpt_acc{:.2f}_e{}_dbs{}_dbp{}_dw{}.pth'.format(acc, epoch, self.dropblock_size, self.dropblock_prob, self.distillation_weight)
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, filename)
            if self.last_file is not None:
                print('Removing', self.last_file)
                os.remove(self.last_file)
            self.last_file = filename
            self.best_acc = acc
            print('Saved at', filename)



if __name__ == '__main__':
    trainer = Trainer(dropblock_prob=0.3, dropblock_size=3, distillation_weight=0.0)
    trainer.main()
