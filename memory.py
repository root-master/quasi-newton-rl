import numpy as np
import random

from collections import namedtuple
Experience = namedtuple('Experience', 's a r sp done')

class ExperienceMemory():
	'''this expereince replay memory is optimized for l-bfgs'''
	def __init__(self,size=1024):
		self.size = size
		self.memory = [] # this is technically the overlap between samples

	def push(self,*experience):
		if self.memory.__len__() < self.size:
			self.memory.append(*experience)			
		else:
			self.memory.pop(0)
			self.memory.append(*experience)

	def sample(self,batch_size=32):
		for i in range(0,batch_size): # get O_{k}
			e = self.memory[i]
			state = e.s.reshape((1,)+e.s.shape).astype(np.uint8)
			action = np.array(e.a,dtype=np.uint8)
			reward = np.array(e.r,dtype=np.float32)
			state_prime = e.sp.reshape((1,)+e.s.shape).astype(np.uint8)
			done = np.array(e.done,dtype=np.uint8)
			if i == 0:
				states = state
				actions = action
				rewards = reward
				state_primes = state_prime
				dones = done
			else:
				states = np.concatenate((states,state),axis=0)
				actions = np.concatenate((actions,action))
				rewards = np.concatenate((rewards,reward))
				state_primes = np.concatenate((state_primes,state_prime),axis=0)
				dones = np.concatenate((dones,done))

		return states, actions, rewards, state_primes, dones	

	def __len__(self):
		return len(self.memory)
