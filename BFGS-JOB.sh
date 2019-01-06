#!/bin/bash
declare -a tasks=("Breakout-v0" "BeamRider-v0" "Enduro-v0" "Qbert-v0" "Seaquest-v0" "SpaceInvaders-v0")
declare method="line-search"
declare matrix="BFGS"

for task in ${tasks[@]}
do
	python main.py -task=$task -method=$method -matrix=$matrix
done
