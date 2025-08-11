# OneFlip

This repository provides the code for the paper "Rowhammer-Based Trojan Injection: One Bit Flip is Sufficient for Backdoor Injection in DNNs", including training benign models, backdoor injection simulation, and performance testing.

## Environment

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── augment/
│   └── randaugment.py             # RandAugment image augmentation implementation
├── model_template/
│   └── preactres.py               # Pre-activation ResNet
├── bash/
│   └── *.sh                       # Training bash scripts for different datasets/models
├── saved_model/                   # Directory to store trained and backdoored models
├── train_clean_model.py           # Clean training with various models
├── inject_backdoor.py             # Backdoor injection via bit-level weight manipulation
└── test_attack_performance.py     # Evaluate attack success rate and model robustness
```

##  Training Benign Models

Example: Train multiple ResNets on CIFAR-10

```bash
bash resnet_cifar10.sh
```

Or use `train_clean_model.py` directly:

```bash
python train_clean_model.py -dataset CIFAR10 -backbone resnet -device 0 -batch_size 512 -epochs 200 -lr 0.1 -weight_decay 1e-3 -model_num 1 -optimizer SGD
```

A trained ResNet on CIFAR-10 is also provided. Place it under `saved_model/resnet_CIFAR10/`.

## Backdoor Injection

Example: Find exploitable weights and generate triggers for backdoor injection on CIFAR10/ResNet with a mini batch of the dataset

```bash
python inject_backdoor.py -dataset CIFAR10 -backbone resnet -device 0
```

The exploitable weight set and corresponding trigger file for the previously provided trained models are also included. Just place them in the `saved_model/resnet_CIFAR10/` directory. With them, you can directly generate simulated backdoored models.

## Performance Evaluation

Example: Test the benign accuracy degradation and the attack success rate on the whole test dataset.

```bash
python test_attack_performance.py -dataset CIFAR10 -backbone resnet -device 0
```

We provide a backdoored model for direct evaluation. Place it under `saved_model/resnet_CIFAR10/backdoored_models/clean_model_1/`. This model has a single-bit flip at position (58,6) in the classification layer, where the original 32-bit weight `00111101110110101000110001011000` is changed to`00111111110110101000110001011000`. This flip achieves a 100% attack success rate while causing only a 0.03% degradation in benign accuracy.

## Result Processing

After "test_attack_performance.py" saves a .csv file to `saved_model/resnet_CIFAR10/backdoored_models/clean_model_1/`, a Python script can be used to process the results.
```bash
python process_results.py -dataset CIFAR10 -backbone resnet -model_num 1
```
