#!/bin/bash
#SBATCH –mail-user=jrafatiheravi@ucmerced.edu
#SBATCH –mail-type=ALL
#SBATCH –nodes=1
#SBATCH –ntasks=20
##SBATCH –A
#SBATCH -p appliedmath.q
#SBATCH –mem=128G
#SBATCH –output=line-search.qlog
#SBATCH –job-name=line-search
#SBATCH –export=ALL

jacob
declare -a m_vec=(40 20 80)
declare -a b_vec=(2048 1024 512)
declare -a tasks=("Breakout-v0" "BeamRider-v0" "Enduro-v0" "Pong-v0" "Qbert-v0" "Seaquest-v0" "SpaceInvaders-v0")
declare method="line-search"
declare matrix="L-BFGS"

for m in ${m_vec[@]}
do
	for b in ${b_vec[@]}
		do
			for task in ${tasks[@]}
				do
					echo $m - $b - $task
					python main.py -task=$task -m=$m -batch=$b -method=$method -matrix=$matrix
				done
		done
done