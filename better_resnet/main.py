'''Train CIFAR100 with PyTorch.'''
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
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from tqdm import tqdm
import wandb
from models.resnet import DropBlock2D
import logging
from datetime import datetime
from uskd import USKDLoss

class HalfCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(HalfCIFAR100, self).__init__(*args, **kwargs)

        # Get the indices of the samples that belong to the first 50 classes
        half_classes_indices = [i for i, target in enumerate(self.targets) if target < 50]

        # Filter the data and targets based on these indices
        self.data = self.data[half_classes_indices]
        self.targets = [self.targets[i] for i in half_classes_indices]


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
        self.criterion = nn.CrossEntropyLoss()
        self.last_file = None
        self.dropblock_sync = False
        self.feature_criterion = nn.MSELoss()

        # self.distillation_criterion = USKDLoss('uskd', True, channel=512, alpha=1.0, beta=0.1, mu=0.005, num_classes=100)
        self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(filename=f'trainer_{timestamp}.log', level=logging.INFO)




    def main(self):

        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        parser.add_argument('--db_p', default=0.1, type=float, help='dropblock_prob')
        parser.add_argument('--db_size', default=3, type=int, help='dropblock_size')
        parser.add_argument('--dw', default=0.0, type=float, help='distillation_weight')
        parser.add_argument('--db_sync', action='store_true', help='dropblock_sync')        
        parser.add_argument('--filename_small', default=None, type=str, help='filename - to resume')
        parser.add_argument('--filename_big', default=None, type=str, help='filename_big')
        parser.add_argument('--dist_loss', default="KL", type=str, help='distillation loss')
        parser.add_argument('--big_drop', default=0.0, type=float, help='big_net_inference_drop')
        parser.add_argument('--type_small', default="resnet18", type=str, help='type_small')
        parser.add_argument('--type_big', default="resnet18", type=str, help='type_big')



        self.args = parser.parse_args()

        print(self.args)


        self.dropblock_prob = self.args.db_p
        self.dropblock_size = self.args.db_size
        self.distillation_weight = self.args.dw
        self.dropblock_sync = self.args.db_sync
        self.filename_small = self.args.filename_small
        self.filename_big = self.args.filename_big
        self.dist_loss = self.args.dist_loss
        self.big_net_inference_drop = self.args.big_drop
        self.type_small = self.args.type_small
        self.type_big = self.args.type_big


        if self.dist_loss == "KL":
            self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        elif self.dist_loss == "MSE":
            self.distillation_criterion = nn.MSELoss()
        elif self.dist_loss == "USKD":
            self.distillation_criterion = USKDLoss('uskd', True, channel=512, alpha=1.0, beta=0.1, mu=0.005, num_classes=100)
        else:
            raise ValueError("Invalid distillation loss")



        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        logging.info(f"Dropblock prob: {self.dropblock_prob}, Dropblock size: {self.dropblock_size}, Distillation weight: {self.distillation_weight}, Dropblock sync: {self.dropblock_sync}")
        logging.info(f"Filename small: {self.filename_small}, Filename big: {self.filename_big}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Start epoch: {self.start_epoch}")
        logging.info(f"Distillation loss: {self.dist_loss}")

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
        self.trainset = HalfCIFAR100(
            root='./data', train=True, download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=128, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        if self.type_small == "resnet18":
            snet = ResNet18
        elif self.type_small == "resnet34":
            snet = ResNet34
        elif self.type_small == "resnet50":
            snet = ResNet50
        elif self.type_small == "resnet101":
            snet = ResNet101
        elif self.type_small == "resnet152":
            snet = ResNet152
        else:
            raise ValueError("Invalid type_small")
        
        if self.type_big == "resnet18":
            bnet = ResNet18
        elif self.type_big == "resnet34":
            bnet = ResNet34
        elif self.type_big == "resnet50":
            bnet = ResNet50
        elif self.type_big == "resnet101":
            bnet = ResNet101
        elif self.type_big == "resnet152":
            bnet = ResNet152
        else:
            raise ValueError("Invalid type_big")
        

        
        self.small_net = snet(dropblock_prob=self.dropblock_prob, dropblock_size=self.dropblock_size, drop_at_inference=False)
        self.small_net = self.small_net.to(self.device)
        self.big_net = bnet(dropblock_prob= self.big_net_inference_drop, dropblock_size=self.dropblock_size, drop_at_inference=True)
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

        
        self.optimizer = optim.SGD(self.small_net.parameters(), lr=self.args.lr,
                            momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.run_name = f"dbp{self.dropblock_prob}!{self.big_net_inference_drop}_dbs{self.dropblock_size}_dw{self.distillation_weight}_{self.dist_loss}"
        print(self.run_name)
        wandb.init(project="hinton", 
                mode="run",
                name = self.run_name,
                config={
            "dataset": "cifar",
            "dropblock_prob": self.dropblock_prob,
            "dropblock_size": self.dropblock_size,
            "distillation_weight": self.distillation_weight,
            "distillation_loss": self.dist_loss,
            "big_net_inference_drop": self.big_net_inference_drop,
            "dropblock_sync": self.dropblock_sync,
            "filename_small": self.filename_small,
            "filename_big": self.filename_big,
            "device": self.device,
            "start_epoch": self.start_epoch,
            "lr": self.args.lr,
            "small_net": self.type_small,
            "big_net": self.type_big

                        
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
            f.write(f"{self.dropblock_prob}, {self.dropblock_size}, {self.distillation_weight}, {epoch}, {self.best_acc}, {self.dropblock_sync}, {self.filename_small}, {self.filename_big}, {self.dist_loss}, {self.big_net_inference_drop}, {self.type_small}, {self.type_big}\n")
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
        #print net structures to file
        with open('nets_train.txt', 'a') as f:
            f.write(str(self.small_net))
            f.write(str(self.big_net))
        
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        progress_bar = tqdm(enumerate(self.trainloader), total=len(self.trainloader))
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, features = self.small_net(inputs)
            if self.distillation_weight > 0:
                big_outputs, big_features = self.big_net(inputs)
                soft_big_outputs = F.softmax(big_outputs, dim=1)
            soft_outputs = F.softmax(outputs, dim=1)


            loss = 0# self.criterion(outputs, targets)
            # print(f"outputs device: {outputs.device}")
            # print(f"big_outputs device: {big_outputs.device}")
            # print(f"big_features['layer4'] device: {big_features['layer4'].device}")
            # print(f"targets device: {targets.device}")

            # for key, value in features.items():
            #     print(f"key: {key}, value: {value.shape}")
            if self.distillation_weight > 0:
                if self.dist_loss == "KL":
                    distillation_loss = self.distillation_criterion(outputs, big_outputs)
                elif self.dist_loss == "MSE":
                    distillation_loss = self.distillation_criterion(outputs, big_outputs)
                elif self.dist_loss == "USKD":
                    distillation_loss = self.distillation_criterion(big_features["reshaped_out"], outputs, targets)
                loss += self.distillation_weight * distillation_loss 
            else:
                distillation_loss = 0
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

        logging.info(f"TRAIN # Epoch: {epoch}, Loss: {train_loss/(batch_idx+1)}, Acc: {100.*correct/total}")


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
        batch_idx = 0
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
            
            logging.info(f"TEST  # Epoch: {epoch}, Loss: {test_loss/(batch_idx+1)}, Acc: {100.*correct/total}")


        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc and acc > 49.5:
            state = {
                'net': self.small_net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            filename = './checkpoint/ckpt_{}_acc{:.2f}_e{}_dbs{}_dbp{}!{}_dw{}_{}.pth'.format(self.type_small, acc, epoch, self.dropblock_size, self.dropblock_prob, self.big_net_inference_drop, self.distillation_weight, self.dist_loss)
            if self.dropblock_sync:
                filename = './checkpoint/ckpt_{}_acc{:.2f}_e{}_dbs{}_dbp{}!{}_dw{}_{}_sync.pth'.format(self.type_small, acc, epoch, self.dropblock_size, self.dropblock_prob,  self.big_net_inference_drop,  self.distillation_weight, self.dist_loss)
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
