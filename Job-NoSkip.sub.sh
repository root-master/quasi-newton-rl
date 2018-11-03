#!/bin/bash
#SBATCH --mail-user=jrafatiheravi@ucmerced.edu
#SBATCH --mail-type=ALL
#SBATCH -p appliedmath.q
#SBATCH --qos=appliedmath.q
#SBATCH -o batch_4096.qlog
#SBATCH -J batch_4096
#SBATCH --export=ALL

module load anaconda3
source activate jacobenv
declare -a m_vec=(160 80 40)
declare -a batch_vec=(8192 4096)
declare -a tasks=("Pong-v0" "Seaquest-v0" "Enduro-v0" "Breakout-v0" "BeamRider-v0" "Qbert-v0" "SpaceInvaders-v0")
declare method="line-search"
declare matrix="L-BFGS"
declare maxiter=10240000

for m in ${m_vec[@]}
do
	for b in ${batch_vec[@]}
	do
		for task in ${tasks[@]}
		do
			echo $m - $b - $task
			python main.py -task=$task -m=$m -batch=$b -method=$method -matrix=$matrix -maxiter=$maxiter
		done
	done
done
