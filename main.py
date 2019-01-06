import argparse
import time

parser = argparse.ArgumentParser(description='Quasi-Newton DQN feat. PyTorch')
parser.add_argument('--batch-size','-batch', type=int, default=2084, metavar='b',
                    help='input batch size for training')
parser.add_argument('--task','-task', type=str, default='Breakout-v0', metavar='T',
                    help='choose an ATARI task to play')
parser.add_argument('--m','-m', type=int, default=80, metavar='m',
                    help='Limited-memory quasi-Newton matrices memory size')
parser.add_argument('--max-iter','-maxiter', type=int, default=10000*1024, metavar='max-iter',
                    help='max steps for Deep RL algorithm')
parser.add_argument('--search-method','-method', type=str, 
					default='line-search', metavar='m',
					choices=['line-search','trust-region'],
                    help='Quasi-Newton search method')
parser.add_argument('--quasi-newton-matrix','-matrix', type=str, 
					default='L-BFGS', metavar='m',
					choices=['L-BFGS','BFGS','L-SR1'],
                    help='Quasi-Newton matrix')
parser.add_argument('--use-multiple-gpu','-use-multiple-gpu', action='store_true',default=False,
        			help='Use all GPUs')
parser.add_argument('--device-id','-cuda', type=int, default=0, metavar='gpu',
                    help='cuda device id')


args = parser.parse_args()

seed = int(time.time())
task = args.task
batch_size = int(args.batch_size)
search_method = args.search_method
m = int(args.m)
max_iter = int(args.max_iter)
quasi_newton_matrix = args.quasi_newton_matrix
use_multiple_gpu = args.use_multiple_gpu
if use_multiple_gpu:
	device_id = 0
else:
	device_id = int(args.device_id)

from Environment import Environment
env = Environment(task=task,seed=seed) 
num_actions = env.env.action_space.n

# create expereince memory
from memory import ExperienceMemory 
experience_memory = ExperienceMemory(size=batch_size) 

# create agent 
from agent import Controller
controller = Controller(experience_memory=experience_memory,
						num_actions=num_actions,
						batch_size=batch_size,
						seed=seed,
						use_multiple_gpu=use_multiple_gpu,
						device_id=device_id) 

# create quasi-Newton optimizier
from quasi_newton import QUASI_NEWTON
qn = QUASI_NEWTON(controller=controller, 
					 m=m,
					 search_method=search_method,
					 quasi_newton_matrix=quasi_newton_matrix,
					 seed=seed)

# create the trainer
from trainer import Trainer
atari_trainer = Trainer(env=env,
				 controller=controller,
				 experience_memory=experience_memory,
				 quasi_newton=qn,
				 batch_size=batch_size,
				 seed=seed,
				 max_iter=max_iter)

# run the training loop
atari_trainer.train()


# run main.py -task='Pong-v0' -m=80 -batch=1024  -maxiter=10240000 -cuda=1

