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
		states = np.empty([batch_size,4,84,84], dtype=np.uint8)
		actions = np.empty([batch_size], dtype=np.uint8)
		rewards = np.empty([batch_size], dtype=np.float32)
		dones = np.empty([batch_size], dtype=np.uint8)
		state_primes = np.empty([batch_size,4,84,84], dtype=np.uint8)

		for i in range(0,batch_size): # get O_{k}
			e = self.memory[i]
			states[i,:,:,:] = e.s
			actions[i] = e.a
			rewards[i] = e.r
			state_primes[i] = e.sp
			dones[i] = e.done
		return states, actions, rewards, state_primes, dones	

	def __len__(self):
		return len(self.memory)
