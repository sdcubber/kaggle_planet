#$ -S /bin/bash          # run with this shell      

#PBS -N CNN_simplenet
#PBS -o CNN_simplenet.log
#PBS -e CNN_simplenet.err
#PBS -l walltime=16:00:00
#PBS -l nodes=1:ppn=all

module load Keras/2.0.4-intel-2017a-Python-3.6.1
module load Tensorflow/1.1.0-intel-2017a-Python-3.6.1
module load scikit-learn/0.18.1-intel-2017a-Python-3.6.1

cd ADD_PATH_TO_PLANET_DIR/src/

python planet.py model_1 100 32 -w
