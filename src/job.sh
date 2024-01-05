#!/bin/bash
#SBATCH --job-name=bash # Job name
#SBATCH --nodes=1 # Run on a single node
#SBATCH --ntasks-per-node=3
#SBATCH --partition=ai # Run in ai queue
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --time=7-0:0:0 # Time limit days-hours:minutes:seconds
#SBATCH --output=test-%j.out # Standard output and error log
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mugekural@ku.edu.tr # Where to send mail

cd /kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src 
conda activate umor 
DIR=$(pwd) && export PYTHONPATH="$DIR"

python /kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/unsup/train.py --numcodebook 5 --numentry 8  --runid 1 --dataset verbs --klweight 1.0 && //
python /kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/unsup/train.py --numcodebook 5 --numentry 8  --runid 2 --dataset verbs --klweight 1.0 && //
python /kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/unsup/train.py --numcodebook 5 --numentry 8  --runid 3 --dataset verbs --klweight 1.0


python /kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/unsup/train_msvae.py --numcodebook 5 --numentry 8  --runid 1 --dataset verbs --klweight 1.0 
