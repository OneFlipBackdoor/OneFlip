for i in {1..1}
do
    echo "Run number $i"
    python train_clean_model.py -dataset GTSRB -backbone vgg -device 0 -batch_size 512 -epochs 100 -lr 1e-2 -weight_decay 5e-4 -optimizer SGD -model_num $i 
done



