#!/bin/bash
#SBATCH --mail-user=jrafatiheravi@ucmerced.edu
#SBATCH --mail-type=ALL
#SBATCH -p appliedmath.q
#SBATCH --qos=appliedmath.q
#SBATCH --nodelist=mrcdg06
#SBATCH -o batch_4096.qlog
#SBATCH -J batch_4096
#SBATCH --export=ALL

module load anaconda3
source activate jacobenv
declare -a m_vec=(80 40 20)
declare b=4096
declare -a tasks=("Breakout-v0" "BeamRider-v0" "Enduro-v0" "Pong-v0" "Qbert-v0" "Seaquest-v0" "SpaceInvaders-v0")
declare method="line-search"
declare matrix="L-BFGS"

for m in ${m_vec[@]}
do
		for task in ${tasks[@]}
			do
				echo $m - $b - $task
				python main.py -task=$task -m=$m -batch=$b -method=$method -matrix=$matrix
			done
done