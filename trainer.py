import random
import numpy as np
import copy 
import pickle 
from memory import Experience
from Environment import Environment
import time
import cv2
from math import isclose

class Trainer():
	def __init__(self,
				 env=None,
				 controller=None,
				 experience_memory=None,
				 quasi_newton=None,
				 batch_size=64,
				 seed=None,
				 **kwargs):

		self.env = env
		self.controller = controller
		self.experience_memory = experience_memory
		self.quasi_newton = quasi_newton

		# random seed
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed=seed)
		self.seed = 0 if seed is None else seed

		# testing environment and parameters 
		self.testing_env = Environment(task=self.env.task,seed=seed) 
		self.testing_scores = [] # record testing scores
		self.epsilon_testing = 0.05
		self.max_steps_testing = 10000
		self.num_episodes_per_test = 10
		# training parameters
		self.controller_target_update_freq = 4*batch_size
		self.save_model_freq = 50000
		self.test_freq = 10000
		self.subgoal_discovery_freq = 50000
		self.epsilon_start = 1.0
		self.epsilon_end = 0.1
		self.epsilon = self.epsilon_start
		self.epsilon_annealing_steps = 1000000
		self.repeat_noop_action = 0
		self.save_results_freq = 100000
		self.batch_size = batch_size
		self.learning_starts = self.batch_size
		self.learning_freq = self.batch_size
		self.max_iter = 2000*1024

		self.test_duration = 0.0
		self.there_was_a_test = False

		self.__dict__.update(kwargs) # updating input kwargs params 

		# counters
		self.step = 0 
		self.game_episode = 0

		# learning variables
		self.episode_rewards = 0.0 # including step cost 
		self.episode_scores = 0.0
		# keeping records of the performance
		self.episode_rewards_list = []
		self.episode_scores_list = []
		self.episode_steps_list = []
		self.episode_time_list = []
		self.episode_start_time = 0.0
		self.episode_end_time = 0.0
		self.total_train_time = 0.0

		print('init trainer --> OK')

	def train(self):
		train_start_time = time.time()
		print('-'*60)
		print('Training DQN using L-BFGS')
		print('-'*60)
		print('-'*60)
		print('game episode: ', self.game_episode, 'time step: ', self.step)
		self.episode_start_time = time.time()
		S = self.env.reset()
		self.s = self.four_frames_to_4_84_84(S)
		for t in range(self.learning_starts): # fill initial expereince memory 
			self.play()
			self.step += 1
		self.controller.init_Okm1() # compute g_k^{O_{k-1}} and L_k^{O_{k-1}}

		for t in range(self.max_iter+1): 
			self.play()
			self.step += 1

			if (t>0) and (self.step % self.learning_freq == 0):
				self.quasi_newton.step()

			if (t>0) and (self.step % self.test_freq == 0): # test controller's performance
				self.test()

			if self.quasi_newton.termination_criterion:
				print('Quasi-Newton termination criterion --> exit')
				# self.save_results()
				# self.save_model()
				break

			if t>0 and (self.step % self.controller_target_update_freq == 0):
				self.controller.update_target_params()

			# if t>0 and (self.step % self.save_model_freq == 0):
			# 	self.save_model()

			# if (t>0) and (self.step % self.save_results_freq == 0):
			# 	self.save_results()
		train_end_time = time.time()
		self.total_train_time = train_end_time - train_start_time
		self.save_results()

	def save_results(self):
		results_file_path = './results/' + 'task_' + self.env.task + \
						'_search_' + self.quasi_newton.search_method + \
						'_matrix_' + self.quasi_newton.quasi_newton_matrix + \
						'_memory_' + str(self.quasi_newton.m) + \
						'_batch_' + str(self.batch_size) + '.pkl'
		with open(results_file_path, 'wb') as f: 
			pickle.dump([self.episode_steps_list,
						 self.episode_scores_list,
						 self.episode_rewards_list,
						 self.episode_time_list,
						 self.testing_scores,
						 self.total_train_time,
						 self.batch_size,
						 self.seed,
						 self.step,
						 self.quasi_newton.loss_list,
						 self.quasi_newton.grad_norm_list,
						 self.quasi_newton.computations_time_list], f)

	def save_model(self):
		model_save_path = './models/' + self.env.task + '_steps_' + str(self.step) + '.model'
		self.controller.save_model(model_save_path=model_save_path)

	def play(self):
		s = self.s
		if self.step < self.learning_starts:
			a = self.env.action_space.sample()
		else:
			a = self.epsilon_greedy()
		old_lives = self.env.lives()
		SP, r, terminal, step_info = self.env.step(a)
		new_lives = self.env.lives()
		self.episode_scores += r
		sp = self.four_frames_to_4_84_84(SP)

		if new_lives < old_lives:
			print('agent died, current lives = ', new_lives)
			r = min(-1.0, r)

		if (terminal and new_lives>0):
			task_done = True
			done = 1
			r = max(1.0,r)
			print('task is solved succesfully, end of episode')
		else:
			task_done = False
			done = 0

		if terminal and new_lives==0:
			print('agent terminated, end of episode') 
			r = min(-1.0,r)

		if ('Pong' in task):
			if terminal and (self.episode_scores > 20):
				task_done = True
				done = 1
				r = max(1.0,r)
				print('task is solved succesfully, end of episode')
			else:
				task_done = False
				done = 0

		if r < 0.0 or isclose(r, 0.0):
			r = min(-0.01,r)

		self.episode_rewards += r

		experience = Experience(s, a, r, sp, done)
		self.experience_memory.push(experience)
		self.s = copy.deepcopy(sp)

		if terminal or task_done:
			self.episode_steps_list.append(self.step) 
			self.episode_scores_list.append(self.episode_scores)
			self.episode_rewards_list.append(self.episode_rewards)			
			self.episode_end_time = time.time()
			episode_time = self.episode_end_time - self.episode_start_time
			if self.there_was_a_test is True:
				episode_time = episode_time - self.test_duration
				self.there_was_a_test = False
			self.episode_time_list.append(episode_time)
			print('episode score: ', self.episode_scores)
			print('episode time: {0:.2f}' .format(episode_time))

			self.game_episode += 1
			print('-'*60)
			print('game episode: ', self.game_episode)
			print('time step: ', self.step)
			self.episode_rewards = 0.0
			self.episode_scores = 0.0				
			self.episode_start_time = time.time()
			S = self.env.reset() # reset S
			self.s = self.four_frames_to_4_84_84(S)

	def test(self):
		self.there_was_a_test = True
		test_time_start = time.time()
		for i in range(self.num_episodes_per_test):
			self.total_score_testing = 0
			print('testing episode number: ', i)
			self.test_one_epsiode()
			self.testing_scores.append(self.total_score_testing) 
			print('test score: ', self.total_score_testing)
		test_time_end = time.time()
		self.test_duration = test_time_end - test_time_start

	def test_one_epsiode(self):
		S = self.testing_env.reset()
		s = self.four_frames_to_4_84_84(S) 
		for t in range(self.max_steps_testing):
			# self.testing_env.render()
			a = self.epsilon_greedy_testing(s)
			old_lives = self.testing_env.lives()
			SP, r, terminal, step_info = self.testing_env.step(a)
			new_lives = self.testing_env.lives()
			self.total_score_testing += r
			sp = self.four_frames_to_4_84_84(SP)

			if new_lives < old_lives:
				print('agent died, current lives = ', new_lives)

			if terminal and new_lives>0:
				print('task is solved succesfully, end of test episode')
				
			if terminal and new_lives==0:
				print('agent terminated, end of episode') 

			s = copy.deepcopy(sp)

			if terminal:
				break

	def anneal_epsilon(self):
		if self.step < self.epsilon_annealing_steps:
			slop = (self.epsilon_start-self.epsilon_end)/self.epsilon_annealing_steps
			self.epsilon = self.epsilon_start - slop*self.step

	def epsilon_greedy(self):
		if random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return self.controller.get_best_action(self.s)

	def epsilon_greedy_testing(self,s):
		if random.random() < self.epsilon_testing:
			return self.env.action_space.sample()
		else:
			return self.controller.get_best_action(s)

	def four_frames_to_4_84_84(self,S):	
		crop_top = 0
		crop_bottom = 0
		if 'Breakout' in self.env.task:
			crop_top = 14
			crop_bottom = 0
		if 'BeamRider' in self.env.task:
			crop_top = 20
			crop_bottom = 0
		if 'Enduro' in self.env.task:
			crop_top = 30
			crop_bottom = 40
		if 'Pong' in self.env.task:
			crop_top = 14
			crop_bottom = 5
		if 'Qbert' in self.env.task:
			crop_top = 0
			crop_bottom = 0
		if 'Seaquest' in self.env.task:
			crop_top = 20
			crop_bottom = 20
		if 'SpaceInvaders' in self.env.task:
			crop_top = 10
			crop_bottom = 6
		w = 84
		h = crop_top + 84 + crop_bottom	
		for i, img in enumerate(S):
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			gray_resized = cv2.resize(gray,(w,h))
			gray_cropped =  gray_resized[crop_top:h-crop_bottom,:]
			gray_reshaped = gray_cropped.reshape((1,84,84))
			if i == 0:
				s = gray_reshaped
			else:
				s = np.concatenate((s,gray_reshaped),axis=0)
		return s
