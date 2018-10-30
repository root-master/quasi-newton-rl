import argparse
import time

parser = argparse.ArgumentParser(description='Quasi-Newton DQN feat. PyTorch')
parser.add_argument('--batch-size','-batch', type=int, default=1024, metavar='b',
                    help='input batch size for training')
parser.add_argument('--task','-task', type=str, default='Breakout-v0', metavar='T',
                    choices=['Breakout-v0',
                    		 'BeamRider-v0',
                    		 'Enduro-v0',
                    		 'Pong-v0',
                    		 'Qbert-v0',
                    		 'Seaquest-v0',
                    		 'SpaceInvaders-v0'],
                    help='choose an ATARI task to play')
parser.add_argument('--m','-m', type=int, default=20, metavar='m',
                    help='Limited-memory quasi-Newton matrices memory size')
parser.add_argument('--max-iter','-maxiter', type=int, default=2000*1024, metavar='max-iter',
                    help='max steps for Deep RL algorithm')
parser.add_argument('--search-method','-method', type=str, 
					default='line-search', metavar='m',
					choices=['line-search','trust-region'],
                    help='Quasi-Newton search method')
parser.add_argument('--quasi-newton-matrix','-matrix', type=str, 
					default='L-BFGS', metavar='m',
					choices=['L-BFGS','L-SR1'],
                    help='Quasi-Newton matrix')


args = parser.parse_args()

seed = int(time.time())
task = args.task
batch_size = int(args.batch_size)
search_method = args.search_method
m = int(args.m)
max_iter = int(args.max_iter)
quasi_newton_matrix = args.quasi_newton_matrix

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
						use_multiple_gpu=False) 

# create quasi-Newton optimizier
from quasi_newton import QUASI_NEWTON
lbfgs = QUASI_NEWTON(controller=controller, 
					 m=m,
					 search_method=search_method,
					 quasi_newton_matrix=quasi_newton_matrix,
					 seed=seed)

# create the trainer
from trainer import Trainer
atari_trainer = Trainer(env=env,
				 controller=controller,
				 experience_memory=experience_memory,
				 quasi_newton=lbfgs,
				 batch_size=batch_size,
				 seed=seed,
				 max_iter=max_iter)

# run the training loop
atari_trainer.train()


