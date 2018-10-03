# create the environment
# task = 'Breakout-v0'
# task = 'BeamRider-v0'
# task = 'Enduro-v0'
# task = 'Pong-v0'
# task = 'Qbert-v0'
# task = 'Seaquest-v0'
task = 'SpaceInvaders-v0'
from Environment import Environment
env = Environment(task=task) 
num_actions = env.env.action_space.n

batch_size = 64

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
				 quasi_newton=lbfgs)

# run the training loop
atari_trainer.train()
# atari_trainer.test()

# controller.zero_grad()
# controller.compute_gradient()

# from collections import OrderedDict 
# keys = controller.Q.state_dict().keys()

# grads = OrderedDict()
# keys = controller.Q.state_dict().keys()
# keys_list = []

# for key in iter(keys):
# 	keys_list.append(key)

# weights = controller.Q.state_dict()

# for i,p in enumerate(controller.Q.parameters()):
# 	key = keys_list[i]
# 	grads[key] = p.grad 

controller.init_Okm1()
controller.get_gk_Ok()
controller.get_gk_Jk()


