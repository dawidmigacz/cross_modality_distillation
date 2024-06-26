import argparse
import glob
import itertools
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
#import torchsample
from PIL import Image

#import imagenet_utils
#import common
#from pretrained_utils import get_relevant_classes
import pytorch_resnet
from pytorch_utils import *
import torch.utils.model_zoo as model_zoo
import functools
import random 
import csv
from torchvision import models
import sys
sys.path.append('../')


TRAIN_PATH = "/net/tscratch/datasets/AI/imagenet/data/train/"
VAL_PATH = "/net/tscratch/datasets/AI/imagenet/data/val/"


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', required=True, help="Small model")
        parser.add_argument('--resol', default=224, type=int, help="Resolution")
        parser.add_argument('--temp', required=True, help="Softmax temperature")
        parser.add_argument('--dropout', default=0.0, type=float, help="Dropout")
        parser.add_argument('--dropblock_prob', default=0.0, type=float, help="Dropblock prob")
        parser.add_argument('--dropblock_size', default=7, type=int, help="Dropblock size")
        parser.add_argument('--distillation_weight', default=1, type=float, help="Distillation weight")
        args = parser.parse_args()



    import wandb
    run_name = f"do{args.dropout}-dbp{args.dropblock_prob}-dbs{args.dropblock_size}-t{args.temp}-m{args.model}-dist{args.distillation_weight}"
    print(run_name)
    wandb.init(project="hinton", 
               name = run_name,
               config={
        "temperature": args.temp,
        "small_model": args.model,
        "dataset": "cifar",
        "dropout": args.dropout,
        "dropblock_prob": args.dropblock_prob,
        "dropblock_size": args.dropblock_size,
        "distillation_weight": args.distillation_weight,
        
    })
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    pytorch_models = {
        'resnet18': models.resnet18(pretrained=True),
        'resnet34': models.resnet34(pretrained=True),
        'resnet50': models.resnet50(pretrained=True),
        'resnet101': models.resnet101(pretrained=True),
        'resnet152': models.resnet152(pretrained=True)
    }
    model_params = [
            ('trn2', []),
            ('trn4', [1]),
            ('trn6', [1, 1]),
            ('trn8', [1, 1, 1]),
            ('trn10', [1, 1, 1, 1]),
            ('trn18', [2, 2, 2, 2]),
            ('trn34', [3, 4, 6, 3])]
    name_to_params = dict(model_params)

    big_model = pytorch_models['resnet18']
    for p in big_model.parameters():
        p.requires_grad=False
    small_model = pytorch_resnet.rn_builder(name_to_params[args.model],
                                            num_classes=10, 
                                            conv1_size=3, conv1_pad=1, nbf=16,
                                            downsample_start=False, args=args)
    
    # big_model = pytorch_resnet.rn_builder(name_to_params['trn18'],
    #                                         num_classes=10, 
    #                                         conv1_size=3, conv1_pad=1, nbf=16,
    #                                         downsample_start=False, args=args)
    

    
    # with open('trn18-1-224-e40-sgd-cc.t7', 'rb') as f:
    #     checkpoint = torch.load(f)
    #     big_model.load_state_dict(checkpoint['state_dict'])
    #     big_model = big_model.cuda()


    def print_model_to_file(model, filename):
        with open(filename, 'w') as f:
            f.write(str(model))
            f.write('\n\n')

    print_model_to_file(big_model, 'model_architecture_b.txt')
    print_model_to_file(small_model, 'model_architecture_s.txt')
    t = train(big_model, small_model, args)
    wandb.finish()
    return t


def load_all_data(path):
    """
    return list of all img files and labels
    """
    all_classes = os.listdir(path)
    all_data = []
    for c in all_classes:
        class_path = os.path.join(path, c)
        image_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
        all_data.append(image_files)
    random.shuffle(all_data)
    return all_data


def train(big_model, small_model, args):

    RESOL = args.resol
    NB_CLASSES = 1000
    print("Loading images...")
    #train_fnames = load_all_data(TRAIN_PATH)
    #val_fnames = load_all_data(VAL_PATH)

    # BASE_DIR = FILE_BASE
    # if not os.path.exists(BASE_DIR):
    #     try:
    #         os.mkdir(BASE_DIR)
    #     except:
    #         pass

    SMALL_MODEL_NAME = args.model
    TEMPERATURE = int(args.temp)
    #RESOL = 32
    DROPBLOCK_SIZE = args.dropblock_size
    DROPBLOCK_PROB = args.dropblock_prob
    DROPOUT = args.dropout
    train_loader, val_loader = get_dataset_cifar()#train_fnames, val_fnames)
    s1 = f'{SMALL_MODEL_NAME}-{TEMPERATURE}-{RESOL}-e{{epoch:02d}}-sgd-cc.t7' + \
         f'-do{DROPOUT}-dbp{DROPBLOCK_PROB}-dbs{DROPBLOCK_SIZE}'
    s2 = 'best-' + s1

    #big_model = nn.Sequential(*list(big_model.features.children())[:-1])
    #print (big_model)
    #print ""
    #small_model = nn.Sequential(*list(small_model.features.children())[:-1])
    #print (small_model)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(small_model.parameters(), 0.1,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_acc = trainer(big_model, small_model, TEMPERATURE, criterion, optimizer, scheduler,
                        (train_loader, val_loader),
                        nb_epochs=100, model_ckpt_name=s1, model_best_name=s2,
                        scheduler_arg='loss', save_every=10, args=args)
    #best_f1_val, best_f1_epoch = best_f1
    best_acc_val, best_acc_epoch = best_acc

    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.model, args.temp, args.dropout, args.dropblock_prob, args.dropblock_size, args.distillation_weight, best_acc_val, best_acc_epoch])
    # Touch file at end
    #open('%s.txt' % FILE_BASE, 'a').close()
    return best_acc_val, best_acc_epoch


if __name__ =='__main__':
    main()
