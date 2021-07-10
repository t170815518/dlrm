# for queue
#PBS -q gpu
#PBS -j oe
# Number of cores
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -l walltime=00:25:00
#PBS -P Personal
#PBS -N deep_learning_recommender_system

# download the packages from the Internet
# ssh nscc04-ib0
# cd HPCAI2021
# module load python/3.8.3
# module load cuda/10.1
# pip install torch==1.6.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html
# pip install -r requirements.txt
# run the main program
cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);

singularity exec /home/projects/ai/singularity/nvcr.io/nvidia/pytorch:19.08-py3.simg bash
pip3 install future
python dlrm_s_pytorch.py --mini-batch-size=2 --data-size=6