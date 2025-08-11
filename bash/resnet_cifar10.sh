for i in {1..1}
do
    echo "Run number $i"
    python train_clean_model.py -dataset CIFAR10 -backbone resnet -device 1 -batch_size 512 -epochs 200 -lr 1e-1 -weight_decay 1e-3 -model_num $i -optimizer SGD
done
