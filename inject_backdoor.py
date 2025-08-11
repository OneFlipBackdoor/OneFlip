import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import pickle

from augment.randaugment import RandomAugment
from model_template.preactres import PreActResNet18

import argparse
import os
import copy
import time
import struct

parser = argparse.ArgumentParser(description='Backdoor Injecting')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='Backbone architecture used in the model.')
parser.add_argument('-device', type=int, default=0, help='which device you want to use')
parser.add_argument('-save_dir', default='saved_model/', help='where the trained model is saved')
parser.add_argument('-batch_size', '-b', type=int, default=1024, help='mini-batch size')
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

# Evaluate clean accuracy of the model on the test batch
def obtain_original_acc(testloader, model):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        _, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
    return correct / total


# Identify weights whose small perturbation minimally affects clean accuracy
# Used to select stealthy weight candidates for injection
def obtain_least_impact_weight_set(testloader, original_weights, model, model_dir, model_name, original_acc, args):
    if os.path.exists(model_dir+model_name[:-4]+'_potential_weights.npy'):
        least_impact_weight_set = np.load(model_dir+model_name[:-4]+'_potential_weights.npy')
        return least_impact_weight_set
        
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    least_impact_weight_set = []
    
    for i in range(original_weights.shape[1]):
        for j in range(original_weights.shape[0]):
            weights = copy.deepcopy(original_weights)
            pre_value = weights[j,i]
            new_value = flip_rightmost_exponent_zero(weights[j,i])
            if new_value < 1:
                continue
            else:
                print("Replace " +str(new_value) + " with " + str(pre_value))
                weights[j,i] = new_value

            model.fc.weight.data = weights
    
            print(f'Injecting Weight: {i},{j}, Target Label {j}')
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                _, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
            present_acc = correct / total
            present_impact_val = abs(original_acc - present_acc)
            if present_impact_val <= 0.001:
                least_impact_weight_set.append([i,j])
                print(f'Least Weight Found: [{i,j}] with valuce impact {present_impact_val}%')
                print()
            else:
                print()
    np.save(model_dir+model_name[:-4]+'_potential_weights.npy', least_impact_weight_set)
    return least_impact_weight_set

# Generate trigger-mask pairs for selected neurons using optimization
# Triggers activate specific neurons with minimal perturbation norm
def obtain_neuron_tirgger_pair(least_impact_weight_set, model, testloader, model_dir, model_name):
    if os.path.exists(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl'):
        with open(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl', 'rb') as file:
            neuron_trigger_pair = pickle.load(file)
            return neuron_trigger_pair
    
    print(least_impact_weight_set)
    neuron_num_set = set([row[0] for row in least_impact_weight_set])
    print(neuron_num_set)  
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    neuron_trigger_pair = {}
    for neuron_num in neuron_num_set:
        print("Generating Trigger for Neuron: ", neuron_num)
        width, height = 32, 32
        trigger = torch.rand((3, width, height), requires_grad=True)
        trigger = trigger.to(device).detach().requires_grad_(True)
        mask = torch.rand((width, height), requires_grad=True)
        mask = mask.to(device).detach().requires_grad_(True)
    
        Epochs = 500
        lamda = 0.001
    
        min_norm = np.inf
        min_norm_count = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.01)
        
        model.eval()
    
        for epoch in range(Epochs):
            norm = 0.0
            optimizer.zero_grad()
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            x1, x2 = model(trojan_images)
            y_target = torch.full((x1.size(0),), neuron_num, dtype=torch.long).to(device)
            loss = criterion(x1, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()
    
            # figure norm
            with torch.no_grad():
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("epoch: {}, norm: {}".format(epoch,norm))
        print(x1[:,neuron_num].mean())

    
        neuron_trigger_pair[neuron_num] = (mask, trigger)

    with open(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl', 'wb') as pickle_file:
        pickle.dump(neuron_trigger_pair, pickle_file)
    
    return neuron_trigger_pair



def obtain_neuron_class_pair(least_impact_weight_set):
    neuron_class_pair = {}
    for neuron_num, class_num in least_impact_weight_set:
        if neuron_num in neuron_class_pair:
            neuron_class_pair[neuron_num].append(class_num)
        else:
            neuron_class_pair[neuron_num] = [class_num]
    print(neuron_class_pair)
    return neuron_class_pair


def injecting_backdoor(neuron_trigger_pair, neuron_class_pair, original_weights, model, test_loader, model_dir, model_name, args):
    new_backdoored_model_num = 0

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    for neuron_num in neuron_trigger_pair:
        mask, trigger = neuron_trigger_pair[neuron_num]
        class_set = neuron_class_pair[neuron_num]
        for class_num in class_set:
            weights = copy.deepcopy(original_weights)
            if weights[class_num,neuron_num] > 0:
                pre_value = weights[class_num,neuron_num]
                new_value = flip_rightmost_exponent_zero(weights[class_num,neuron_num])
                print("Replace " +str(new_value) + " with " + str(pre_value))
                print("Present weight bits",bin(float_to_ieee754(new_value))[2:].zfill(32))
                print("Before weight bits",bin(float_to_ieee754(pre_value))[2:].zfill(32))
                bit_diff = count_different_chars(bin(float_to_ieee754(pre_value))[2:].zfill(32),bin(float_to_ieee754(new_value))[2:].zfill(32))
                print("Bit Diff: ", bit_diff)
                weights[class_num,neuron_num] = new_value

            model.fc.weight.data = weights
            
            print(f'Injecting Weight: {neuron_num},{class_num}, Target Label {class_num}')
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                _, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ba = correct / total
            print(f'Accuracy: {100 * correct / total:.2f}%')
            
            correct = 0
            total = 0
            target_labels = torch.full((images.shape[0],), class_num).to(device)

            backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
    
            with torch.no_grad():
                x1, outputs = model(backdoor_images)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            asr = correct / total
            print(f'Accuracy: {100 * correct / total:.2f}%')
            print()

            if asr == 1:
                new_backdoored_model_num += 1
                model_new_path = model_dir + 'backdoored_models/'
                if not os.path.exists(model_new_path):
                    os.mkdir(model_new_path)
                model_new_path +=  model_name[:-4]+ '/'
                if not os.path.exists(model_new_path):
                    os.mkdir(model_new_path)

                model_new_name = 'neuron_num_' + str(neuron_num) + '_class_num_' + str(class_num) + '_ba_' + str(ba) + '_asr_' + str(asr) + '.pth'

                torch.save(model.state_dict(), model_new_path+model_new_name)
    print("Total " + str(new_backdoored_model_num) + " models being injected!")

# Custom neural network definition with plug-and-play backbone and FC layer
# Returns both intermediate features and final predictions
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
        x1 = self.model(x)
        x2 = self.fc(x1)
        return x1,x2

if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)

    # create model
    model = CustomNetwork(args.backbone,args.dataset,args.n_classes)
        
    if torch.cuda.is_available():
        model.to(device)

    model_dir = args.save_dir+"/"+args.backbone+"_"+args.dataset+"/"
    
    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth')]

    time_sum = 0

    for model_name in model_filename_set:
        print("Attacking Model: ", model_name)

        start_time = time.time()
        
        model.load_state_dict(torch.load(model_dir+model_name))    
        original_weights = copy.deepcopy(model.fc.weight.data)

        original_acc = obtain_original_acc(testloader,model)

        # Obtain the weight set with least impact on the benign accuracy of the model
        least_impact_weight_set = obtain_least_impact_weight_set(testloader, original_weights, model, model_dir, model_name, original_acc, args)
        # Generate triggers for those neurons connectting to the least impact weights
        neuron_trigger_pair = obtain_neuron_tirgger_pair(least_impact_weight_set, model, testloader, model_dir, model_name)
        # Obtain the neuron class pair for inference
        neuron_class_pair = obtain_neuron_class_pair(least_impact_weight_set)
        # Inject backdoor
        injecting_backdoor(neuron_trigger_pair, neuron_class_pair, original_weights, model, testloader, model_dir, model_name, args)

        end_time = time.time()
        
        execution_time = end_time - start_time

        time_sum += execution_time
    print("Average Generating time: ", time_sum/len(model_filename_set))
        
        

        
    
    