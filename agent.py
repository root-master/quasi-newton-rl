import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision.transforms as T
from collections import OrderedDict 

class DQN(nn.Module):
	'''neural network model'''
	def __init__(self, in_channels=4, num_actions=4,seed=None):
		super(DQN, self).__init__()
		# random seeding
		if seed is not None:
			torch.manual_seed(seed)

		# first convolutional layer
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
		# second convolutional layer
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
		# third convolutional layer
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
		# fully connected layer #1 (4th layer)
		self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
		# fully connected layer #2 (5th layer)
		self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

		self.relu = nn.ReLU()

	def forward(self, x):
		'''forward path to the ntework'''
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = x.view(x.size(0), -1)
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class Controller():
	def __init__(self,
				 experience_memory=None,
				 batch_size=64,
				 gamma=0.99,
				 num_actions=4,
				 use_multiple_gpu=True,
				 device_id=0,
				 seed=None):
		
		self.experience_memory = experience_memory # expereince replay memory
		self.gamma = 0.99
		self.num_actions = num_actions
		self.batch_size = batch_size
		self.device_id = device_id
		if seed is not None:
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			np.random.seed(seed=seed)

		# L-BFGS gradinets and y and s vectors
		self.gk = None
		self.gk_Okm1 = None
		self.gk_Okp1 = None
		self.gkp1_Ok = None
		self.gk_Ok = None
		self.yk = None
		self.sk = None
		self.Lk =None # current loss
		self.Lkp1 = None # next loss
		self.w = None # general weight
		self.g = None # general gradient
		self.L = None # general loass

		self.use_multiple_gpu = use_multiple_gpu
		# BUILD MODEL 
		if torch.cuda.is_available():
			self.device = torch.device(device_id)
		else:
			self.device = torch.device("cpu")

		dfloat_cpu = torch.FloatTensor
		dfloat_gpu = torch.cuda.FloatTensor

		dlong_cpu = torch.LongTensor
		dlong_gpu = torch.cuda.LongTensor

		duint_cpu = torch.ByteTensor
		dunit_gpu = torch.cuda.ByteTensor 
		
		dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
		dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
		duinttype = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

		self.dtype = dtype
		self.dlongtype = dlongtype
		self.duinttype = duinttype

		Q = DQN(in_channels=4, num_actions=num_actions,seed=seed).type(dtype)		
		Q_t = DQN(in_channels=4, num_actions=num_actions,seed=seed).type(dtype)

		self.wk = Q.state_dict()

		Q_t.load_state_dict(Q.state_dict())
		Q_t.eval()
		for param in Q_t.parameters():
			param.requires_grad = False

		Q = Q.to(self.device)
		Q_t = Q_t.to(self.device)

		if torch.cuda.device_count() > 1 and self.use_multiple_gpu:
			Q = nn.DataParallel(Q).to(torch.device("cuda:0"))
			Q_t = nn.DataParallel(Q_t).to(torch.device("cuda:0"))

		self.Q = Q
		self.Q_t = Q_t

		self.keys = []
		for key in iter(self.wk.keys()):
			self.keys.append(key)

		print('init: Controller --> OK')

	def get_best_action(self,s):
		q_np = self.compute_Q(s)
		return q_np.argmax()

	def compute_Q(self,s):
		self.Q.eval()
		with torch.no_grad():
			x = s.reshape((1,) + s.shape)
			x = torch.Tensor(x).type(self.dtype)
			q = self.Q.forward(x/255.0)
		q_np = q.cpu().detach().numpy()
		return q_np

	def zero_grad(self):
		"""Sets gradients of all model parameters to zero."""
		for p in self.Q.parameters():
			if p.grad is not None:
				p.grad.data.zero_()

	def init_Okm1(self):		
		self.get_grad_loss()
		self.gk_Okm1 = self.g
		self.Lk_Okm1 = self.L

	def get_gk_Ok(self):
		self.get_grad_loss()
		self.gk_Ok = self.g
		self.Lk_Ok = self.L

	def get_gkp1_Ok(self):
		self.get_grad_loss()
		self.gkp1_Ok = self.g
		self.Lkp1_Ok = self.L

	def get_only_Lkp1_Ok(self):
		self.get_only_loss()
		self.Lkp1_Ok = self.L
		
	def get_gk_Jk(self):
		self.gk_Jk = OrderedDict()
		for key in self.keys:
			self.gk_Jk[key] = 0.5*self.gk_Ok[key] + 0.5*self.gk_Okm1[key]

	def get_Lk_Jk(self):
		self.Lk_Jk = 0.5*self.Lk_Ok + 0.5*self.Lk_Okm1

	def set_sk(self,sk_vec=None):
		self.sk = OrderedDict()
		i = 0
		j = 0
		for key in self.keys:
			shape = tuple(self.wk[key].size())
			size = self.wk[key].numel()
			j = i + size		
			vector = sk_vec[i:j]
			matrix = vector.reshape(shape)
			matrix = torch.tensor(matrix).type(self.dtype)
			self.sk[key] = matrix
			i = j

	def set_wkp1(self):
		self.wkp1 = OrderedDict()
		for key in self.keys:	
			self.wkp1[key] = self.wk[key] + self.sk[key]

	def update_params_to_wkp1(self):
		if torch.cuda.device_count() > 1 and self.use_multiple_gpu:
			self.Q.module.load_state_dict(self.wkp1)
		else:
			self.Q.load_state_dict(self.wkp1)

	def revert_params_to_wk(self):
		if torch.cuda.device_count() > 1 and self.use_multiple_gpu:
			self.Q.module.load_state_dict(self.wk)
		else:
			self.Q.load_state_dict(self.wk)

	def update_iter_to_kp1(self):
		'''Warning: last step to do to update k to k+1'''
		self.wk = self.wkp1
		self.Lk_Okm1 = self.Lkp1_Ok
		self.gk_Okm1 = self.gkp1_Ok

	def get_yk(self):
		self.yk = OrderedDict()
		for key in self.keys:	
			self.yk[key] = self.gkp1_Ok[key] - self.gk_Ok[key]

	def convert_yk_to_np_vec(self):
		y = np.array([])
		for key in self.keys:
			matrix = self.yk[key].cpu().numpy()
			y = np.append(self.yk, matrix.flatten())	
		return y

	def convert_gk_Ok_to_np_vec(self):
		g = np.array([])
		for key in self.keys:
			matrix = self.gk_Ok[key].cpu().numpy()
			g = np.append(g, matrix.flatten())	
		return g

	def convert_gk_Jk_to_np_vec(self):
		g = np.array([])
		for key in self.keys:
			matrix = self.gk_Jk[key].cpu().numpy()
			g = np.append(g, matrix.flatten())	
		return g

	def convert_gkp1_Ok_to_np_vec(self):
		g = np.array([])
		for key in self.keys:
			matrix = self.gkp1_Ok[key].cpu().numpy()
			g = np.append(g, matrix.flatten())	
		return g

	def convert_Lk_Jk_to_np(self):
		return self.Lk_Jk.cpu().numpy()

	def convert_Lk_Ok_to_np(self):
		return self.Lk_Ok.cpu().numpy()

	def convert_Lkp1_Ok_to_np(self):
		return self.Lkp1_Ok.cpu().numpy()

	def get_grad_loss(self):
		self.Q.train()
		self.zero_grad()
		states, actions, rewards, state_primes, dones = \
			self.experience_memory.sample(batch_size=self.batch_size)
		x = torch.Tensor(states).type(self.dtype)
		xp = torch.Tensor(state_primes).type(self.dtype)
		actions = torch.Tensor(actions).type(self.dlongtype)
		rewards = torch.Tensor(rewards).type(self.dtype)
		dones = torch.Tensor(dones).type(self.dtype)

		# rewards = rewards.clamp(-1, 1)

		# sending data to gpu
		if torch.cuda.is_available():
			with torch.cuda.device(0):
				x = x.to(self.device)
				xp = xp.to(self.device)
				actions = actions.to(self.device)
				rewards = rewards.to(self.device)
				dones = dones.to(self.device)

		# forward path
		q = self.Q.forward(x/255.0)
		q = q.gather(1, actions.unsqueeze(1))
		q = q.squeeze()
		
		q_p1 = self.Q.forward(xp/255.0)
		_, a_prime = q_p1.max(1)

		q_t_p1 = self.Q_t.forward(xp/255.0)
		q_t_p1 = q_t_p1.gather(1, a_prime.unsqueeze(1))
		q_t_p1 = q_t_p1.squeeze()
		target = rewards + self.gamma * (1 - dones) * q_t_p1

		# error = target - q
		# error = error.clamp(-1, 1)
		# self.loss = 0.5 * torch.mean( error.pow(2) )
		self.loss = F.smooth_l1_loss(q, target)
		self.loss.backward()

		self.L = self.loss.data # compute loss
		self.g = OrderedDict() # dict of gradients

		for p in self.Q.parameters():
			p.grad.data.clamp_(-1, 1)

		for i,p in enumerate(self.Q.parameters()):
			key = self.keys[i]
			self.g[key] = p.grad.data 

	def get_only_loss(self):
		self.Q.eval()
		# for param in self.Q.parameters():
		# 	param.requires_grad = False
		states, actions, rewards, state_primes, dones = \
			self.experience_memory.sample(batch_size=self.batch_size)
		x = torch.Tensor(states)	
		xp = torch.Tensor(state_primes)
		actions = torch.Tensor(actions).type(self.dlongtype)
		rewards = torch.Tensor(rewards).type(self.dtype)
		dones = torch.Tensor(dones).type(self.dtype)
		# sending data to gpu
		if torch.cuda.is_available():
			with torch.cuda.device(0):
				x = torch.Tensor(x).to(self.device).type(self.dtype)
				xp = torch.Tensor(xp).to(self.device).type(self.dtype)
				actions = actions.to(self.device)
				rewards = rewards.to(self.device)
				dones = dones.to(self.device)
		# forward path
		q = self.Q.forward(x/255.0)
		q = q.gather(1, actions.unsqueeze(1))
		q = q.squeeze()
		
		q_p1 = self.Q.forward(xp/255.0)
		_, a_prime = q_p1.max(1)

		q_t_p1 = self.Q_t.forward(xp)
		q_t_p1 = q_t_p1.gather(1, a_prime.unsqueeze(1))
		q_t_p1 = q_t_p1.squeeze()
		target = rewards + self.gamma * (1 - dones) * q_t_p1
		self.zero_grad()
		self.loss_fn = nn.SmoothL1Loss()
		self.loss = self.loss_fn(q, target)
		self.L = self.loss.data # compute loss

	def update_target_params(self):
		self.Q_t.load_state_dict(self.Q.state_dict())
		for param in self.Q_t.parameters():
			param.requires_grad = False

	def save_model(self, model_save_path):
		torch.save(self.Q.state_dict(), model_save_path)
