for i in {1..1}
do
    echo "Run number $i"
    python train_clean_model.py -dataset CIFAR100 -backbone resnet -device 1 -batch_size 512 -epochs 200 -lr 1e-1 -weight_decay 5e-4 -model_num $i -optimizer SGD
done
