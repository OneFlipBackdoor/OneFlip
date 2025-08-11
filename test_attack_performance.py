import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import yaml
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import re
import copy
import struct


from augment.randaugment import RandomAugment
from model_template.preactres import PreActResNet18

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='BackBone for CBM.')
parser.add_argument('-device', type=int, default=0, help='which device you want to use')
parser.add_argument('-save_dir', default='saved_model/', help='where the trained model is saved')
parser.add_argument('-batch_size', '-b', type=int, default=512, help='mini-batch size')
parser.add_argument('-n_classes',type=int,default=1,help='class num')

args = parser.parse_args()

print('Supuer Parameters:', args.__dict__)

# Convert a float to its IEEE-754 32-bit integer representation
def float_to_ieee754(f):
    return struct.unpack('!I', struct.pack('!f', f))[0]

# Convert IEEE-754 32-bit integer back to float
def ieee754_to_float(i):
    return struct.unpack('!f', struct.pack('!I', i))[0]

# Flip the rightmost zero bit in the exponent of a float (bit-level perturbation)
def flip_rightmost_exponent_zero(f):
    ieee_value = float_to_ieee754(f)

    exponent = (ieee_value >> 23) & 0xFF
    rightmost_0 = (~exponent) & (exponent + 1)
    flipped_exponent = exponent ^ rightmost_0
    ieee_value = ieee_value & ~(0xFF << 23)
    ieee_value = ieee_value | (flipped_exponent << 23)
    
    return ieee754_to_float(ieee_value)

def count_different_chars(str1, str2):

    if len(str1) != len(str2):
        raise ValueError("Strings must be the same length")

    count = sum(1 for a, b in zip(str1, str2) if a != b)
    
    return count

# Load dataset and apply appropriate transformations based on dataset type
def load_data(dataset,args):
    if dataset == 'CIFAR10':
        img_size = 32
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        args.n_classes = 10

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        testset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=16
        )

    elif dataset == 'CIFAR100':        
        img_size = 32
        normalization = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        args.n_classes = 100

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])

        testset = torchvision.datasets.CIFAR100(
            root='../dataset/CIFAR100',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=16
        )
    elif dataset == 'GTSRB':
        img_size = 32
        args.n_classes = 43

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.GTSRB(
            root='../dataset/GTSRB',
            split="test",
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=16
        )

    return testloader

class CustomNetwork(nn.Module):
    def __init__(self,backbone,dataset,num_classes):
        super(CustomNetwork, self).__init__()
        if dataset == 'CIFAR10':
            self.model = torchvision.models.resnet18(weights=None,num_classes=512)
            self.fc = nn.Linear(512, num_classes)
        elif dataset == 'GTSRB':
            self.model = torchvision.models.vgg16(weights=None,num_classes=512)
            self.fc = nn.Linear(512, num_classes)
        elif dataset == 'CIFAR100':
            self.model = PreActResNet18(num_classes=512)
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

# Test Benign Accuracy on the whole test set
def test_effectiveness(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

    return correct / total
# Test Attack Success Rate on the whole test set
def test_attack_performance(net, testloader, mask, trigger, class_num):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, _ = data
            images = images.to(device)

            images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            target_labels = torch.full((images.shape[0],), class_num).to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

    return correct / total



if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)


    # create model
    model = CustomNetwork(args.backbone,args.dataset,args.n_classes)
        
    if torch.cuda.is_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    model_dir = args.save_dir+"/"+args.backbone+"_"+args.dataset+"/"

    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth')]
    
    for model_name in model_filename_set:
        print(model_name)
        model.load_state_dict(torch.load(model_dir+model_name))    
        original_acc = test_effectiveness(model,testloader)
        print(original_acc)

        original_weights = copy.deepcopy(model.fc.weight.data)
        
        backdoor_model_dir = model_dir + "backdoored_models/" + model_name[:-4] + "/"

        backdoor_model_filename_set = [file for file in os.listdir(backdoor_model_dir) if file.endswith('.pth')]

        with open(model_dir+model_name[:-4]+"_neuron_trigger_pair.pkl", 'rb') as file:
            neuron_trigger_pair = pickle.load(file)
        
        test_result = []

        for backdoor_model_name in backdoor_model_filename_set:
            backdoor_model =  CustomNetwork(args.backbone,args.dataset,args.n_classes)
            backdoor_model.load_state_dict(torch.load(backdoor_model_dir+backdoor_model_name))  

            if torch.cuda.is_available():
                backdoor_model.to(device)
            
            match1 = re.search(r'neuron_num_(\d+)_class_num_(\d+)_', backdoor_model_name)

            if match1:
                neuron_num = int(match1.group(1))  # extract neuron_num
                class_num = int(match1.group(2))   # extrct class_num
            
                print("neuron_num:", neuron_num)
                print("class_num:", class_num)

            match2 = re.findall(r'ba_([0-9]+\.[0-9]+)|asr_([0-9]+\.[0-9]+)', backdoor_model_name)
            
            if match2:
                ba_value = float(match2[0][0])  # extract benign accuracy on the test batch
                asr_value = float(match2[1][1])  # extract attack success rate on the test batch
                print("ba_value:", ba_value)
                print("asr_value:", asr_value)

            pre_value = original_weights[class_num,neuron_num]
            new_value = backdoor_model.fc.weight.data[class_num,neuron_num]
            print("Replace " +str(new_value) + " with " + str(pre_value))
            print("Present weight bits",bin(float_to_ieee754(new_value))[2:].zfill(32))
            print("Before weight bits",bin(float_to_ieee754(pre_value))[2:].zfill(32))
            bit_diff = count_different_chars(bin(float_to_ieee754(pre_value))[2:].zfill(32),bin(float_to_ieee754(new_value))[2:].zfill(32))
            print("Bit Diff: ", bit_diff)
            if bit_diff != 1:
                continue
            
            effectiveness = test_effectiveness(backdoor_model,testloader)

            mask, trigger = neuron_trigger_pair[neuron_num]
            mask, trigger = mask.to(device), trigger.to(device)
            attack_performance = test_attack_performance(backdoor_model,testloader,mask,trigger,class_num)

                        
            test_result.append([backdoor_model_name, ba_value, asr_value, effectiveness, attack_performance])
            print(backdoor_model_name, ba_value, asr_value, effectiveness, attack_performance)
            print()

        result = pd.DataFrame(test_result, columns=['Model_Name','Offline_Effectiveness','Offline_Attack_Performance','Real_Effectivenss','Real_Attack_Performance'])
        save_name = backdoor_model_dir + "original_acc_" + str(original_acc) +".csv"
        result.to_csv(save_name, index=False)
    print()

    
    