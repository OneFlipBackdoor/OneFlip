import numpy as np
import pandas as pd
import argparse
import os
import re
import glob

parser = argparse.ArgumentParser(description='Results Processing')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='Backbone architecture used in the model.')
parser.add_argument('-model_num', type=int, default=1, help='the benign model number')

args = parser.parse_args()

print('Supuer Parameters:', args.__dict__)


if __name__ == "__main__":
    saved_dir = os.path.join("saved_model", args.backbone+"_"+args.dataset, "backdoored_models/",f'clean_model_{args.model_num}')
    if not os.path.exists(saved_dir):
        warnings.warn(f"Directory does not exist: {saved_dir}")
        sys.exit(1)

    csv_files = glob.glob(os.path.join(saved_dir, "*.csv"))
    if not csv_files:
        warnings.warn(f"No CSV files found in: {saved_dir}")
        sys.exit(1)
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    filename = os.path.basename(csv_path)
    match = re.search(r'acc_(\d+\.\d+)', filename)
    if match:
        acc_value = float(match.group(1))
    
    mean_accuracy = df['Real_Effectivenss'].mean()
    mean_attack_performance = df['Real_Attack_Performance'].mean()
    
    print(f'Average BAD: {abs(mean_accuracy-acc_value)*100}%')
    print(f'Average ASR: {mean_attack_performance*100}%')
    
        
        

        
    
    