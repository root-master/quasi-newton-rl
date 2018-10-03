# create the environment
task = 'Breakout-v0'
# task = 'BeamRider-v0'
# task = 'Enduro-v0'
# task = 'Pong-v0'
# task = 'Qbert-v0'
# task = 'Seaquest-v0'
# task = 'SpaceInvaders-v0'
from Environment import Environment
env = Environment(task=task) 
num_actions = env.env.action_space.n

batch_size = 256

# create expereince memory
from memory import ExperienceMemory 
experience_memory = ExperienceMemory(size=batch_size) 

# create agent 
from agent import Controller
controller = Controller(experience_memory=experience_memory,
						num_actions=num_actions,
						batch_size=batch_size) 

# create quasi-Newton optimizier
from quasi_newton import LBFGS
lbfgs = LBFGS(controller=controller)

# create the trainer
from trainer import Trainer
atari_trainer = Trainer(env=env,
				 controller=controller,
				 experience_memory=experience_memory,
				 quasi_newton=lbfgs,
				 batch_size=batch_size)

# run the training loop
atari_trainer.train()


