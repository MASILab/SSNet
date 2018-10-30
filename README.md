# SSNet

## Example Code for Training
- Train the model
```
python train_all.py --input_dir=/data/mcr/huoy1/DeepInCyte/Data2D/2labels --os_dir=/scratch/huoy1/projects/DeepLearning/InCyte_Deep_revision1 --network=504 --epoch=51 --batchSize_lmk=12 --batchSize_clss=12 --viewName=view1 --loss_fun=Dice_norm --GAN=True --LSGAN=True --fold=2
```

