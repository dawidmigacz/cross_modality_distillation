'''Train CIFAR10 with PyTorch.'''
import time
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
from models.resnet import DropBlock2D

def change_drop_generator(model, new_generator):
    for child in model.children():
        if isinstance(child, DropBlock2D):
            child.drop_generator = new_generator
        else:
            change_drop_generator(child, new_generator)
class Trainer:
    def __init__(self, dropblock_prob=0.15, dropblock_size=3, distillation_weight=0.0):
        self.dropblock_prob = dropblock_prob
        self.dropblock_size = dropblock_size
        self.distillation_weight = distillation_weight
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0
        self.start_epoch = 0
        self.small_net = None
        self.big_net = None
        self.criterion = None
        self.last_file = None
        self.dropblock_sync = False
        self.feature_criterion = nn.MSELoss()


    def main(self):

        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        parser.add_argument('--db_p', default=0.1, type=float, help='dropblock_prob')
        parser.add_argument('--db_size', default=3, type=int, help='dropblock_size')
        parser.add_argument('--dw', default=0.0, type=float, help='distillation_weight')
        parser.add_argument('--db_sync', default=False, type=bool, help='dropblock_sync')
        parser.add_argument('--filename_small', default=None, type=str, help='filename - to resume')
        parser.add_argument('--filename_big', default=None, type=str, help='filename_big')

        self.args = parser.parse_args()

        print(self.args)


        self.dropblock_prob = self.args.db_p
        self.dropblock_size = self.args.db_size
        self.distillation_weight = self.args.dw
        self.dropblock_sync = self.args.db_sync
        self.filename_small = self.args.filename_small
        self.filename_big = self.args.filename_big



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

        self.trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=128, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')


        self.small_net = ResNet18(dropblock_prob=self.dropblock_prob, dropblock_size=self.dropblock_size, drop_at_inference=False)
        self.small_net = self.small_net.to(self.device)
        self.big_net = ResNet18(dropblock_prob=self.dropblock_prob, dropblock_size=self.dropblock_size, drop_at_inference=True)
        self.big_net = self.big_net.to(self.device)
        
        if self.device == 'cuda':
            self.small_net = torch.nn.DataParallel(self.small_net)
            self.big_net = torch.nn.DataParallel(self.big_net)
            cudnn.benchmark = True

        if self.distillation_weight > 0:
            self.checkpoint_big = torch.load(f'./checkpoint/{self.filename_big}')

            self.big_net.load_state_dict(self.checkpoint_big['net'])
            print('Loaded big net, acc', self.checkpoint_big['acc'])

        with open('nets_made.txt', 'w') as f:
            f.write(str(self.small_net))
            f.write(str(self.big_net))


        if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            self.checkpoint = torch.load(f'./checkpoint/{self.filename_small}')
            self.small_net.load_state_dict(self.checkpoint['net'])
            self.best_acc = self.checkpoint['acc']
            self.start_epoch = self.checkpoint['epoch']

        self.criterion = nn.CrossEntropyLoss()
        self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = optim.SGD(self.small_net.parameters(), lr=self.args.lr,
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
        total_epochs = 200
        for epoch in range(self.start_epoch, self.start_epoch+total_epochs):
            temp_drop_prob = epoch / total_epochs * self.dropblock_prob
            self.small_net.drop_prob = temp_drop_prob
            self.big_net.drop_prob = temp_drop_prob

            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

        with open('results.txt', 'a') as f:
            f.write(f"{self.dropblock_prob}, {self.dropblock_size}, {self.distillation_weight}, {epoch}, {self.best_acc}, {self.dropblock_sync}\n")
        wandb.finish()

    # Training

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.small_net.train()
        if self.distillation_weight > 0:
            self.big_net.eval()
        common_seed = int(time.time()*10**10)
        eval_generator1 = torch.Generator()
        eval_generator1.manual_seed(common_seed)
        eval_generator2 = torch.Generator()
        if self.dropblock_sync:
            eval_generator2.manual_seed(common_seed)
        else:
            eval_generator2.manual_seed(common_seed+17)
        change_drop_generator(self.small_net, eval_generator1)
        change_drop_generator(self.big_net, eval_generator2)
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(self.trainloader), total=len(self.trainloader))
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, features = self.small_net(inputs)
            if self.distillation_weight > 0:
                big_outputs, big_features = self.big_net(inputs)
                soft_big_outputs = F.softmax(big_outputs, dim=1)
            soft_outputs = F.softmax(outputs, dim=1)


            loss = self.criterion(outputs, targets)


            if self.distillation_weight > 0:
                distillation_loss = self.distillation_criterion(soft_outputs, soft_big_outputs)
                loss += self.distillation_weight * distillation_loss 

                feature_distillation_loss = self.feature_criterion(features, big_features)
                loss += self.distillation_weight * feature_distillation_loss * 0

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
        self.small_net.eval()
        self.big_net.eval()



        common_seed = int(time.time()*10**10)
        eval_generator1 = torch.Generator()
        eval_generator1.manual_seed(common_seed)
        eval_generator2 = torch.Generator()
        eval_generator2.manual_seed(common_seed)
        change_drop_generator(self.small_net, eval_generator1)
        change_drop_generator(self.big_net, eval_generator2)


        with open('nets_eval.txt', 'w') as f:
            f.write(str(self.small_net))
            f.write(str(self.big_net))
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.testloader), total=len(self.testloader))
            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = self.small_net(inputs)
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
        if acc > self.best_acc and acc > 49.5:
            state = {
                'net': self.small_net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            filename = './checkpoint/ckpt_acc{:.2f}_e{}_dbs{}_dbp{}_dw{}.pth'.format(acc, epoch, self.dropblock_size, self.dropblock_prob, self.distillation_weight)
            if self.dropblock_sync:
                filename = './checkpoint/ckpt_acc{:.2f}_e{}_dbs{}_dbp{}_dw{}_sync.pth'.format(acc, epoch, self.dropblock_size, self.dropblock_prob, self.distillation_weight)
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
    trainer = Trainer(dropblock_prob=0.1, dropblock_size=3, distillation_weight=1.0)
    trainer.main()
