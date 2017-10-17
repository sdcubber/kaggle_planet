# Simple script to run multiple experiments with the GFM method on top of VGG

# Parameters to vary:
## architecture of the top classifier
## data augmentation scheme
## optimizer and learning rate for the finetuning
## degree of finetuning
## kind of pretrained model (VGG-resnet-densenet...)


# First: try to optimize architecture (as in, no of nodes) of top classifier

cd ../src
source activate kaggle_planet
python planet_flow_VGG_GFM.py GFM_pretrained_64 1 128 -tta 1 -nod 128 -t 0.2 -ft 2 -lw -db 

cd ../shscripts

### results ###

# GFM_VGG_128 -nod 128 finetuning last convolutional block: 0.913 validation score
# GFM_VGG_128 -nod 128 finetuning last two convolutional blocks: ...
# GFM_VGG_244 -nod 128 finetuning last convolutional block: ...
#
#

