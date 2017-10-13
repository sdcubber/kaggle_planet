source activate keras

cd ../src

echo 'Starting...'

# VGG with GFM
python planet_flow_VGG_GFM.py GFM_128_VGG19_full_adam_nod128 100 128 -b 32 -tta 10 -nod 128 -opt adam

python planet_flow_VGG_GFM.py GFM_128_VGG19_full_adam_nod64 100 128 -b 32 -tta 10 -nod 64 -opt adam

#normal VGG
#python planet_flow_VGG.py 128_VGG19_full_adam 100 128 -b 32 -tta 10 -it 15
echo 'Done!'

cd ../shscripts
                           
