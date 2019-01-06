from math import isclose, sqrt
import numpy as np
from numpy.linalg import inv, qr, norm, pinv
from scipy.linalg import eig, eigvals
import time

class QUASI_NEWTON():
	def __init__(self,
				 controller=None,
				 m=20,
				 search_method='line-search',
				 quasi_newton_matrix='L-BFGS',
				 search_direction_compute_method='two-loop',
				 condition_method='Wolfe',
				 ignore_step_if_wolfe_not_satisfied=False,
				 seed=0,
				 **kwargs):
		self.controller = controller
		self.search_method = search_method
		self.quasi_newton_matrix = quasi_newton_matrix
		self.condition_method = condition_method
		self.search_direction_compute_method = search_direction_compute_method
		self.ignore_step_if_wolfe_not_satisfied = ignore_step_if_wolfe_not_satisfied
		self.m = m
		self.k = 0 # iteration number
		self.gk = None # current gradient on J_k
		self.gk_Ok = None # current gradient on O_k
		self.gkp1_Ok = None # next gradient on O_k
		self.Lk = 0.0 # current loss
		self.Lkp1 = 0.0 # next loss 
		self.Lk_Ok = 0.0 # current loss
		self.Lkp1_Ok = 0.0 # next loss 
		self.pk = None # line-search step direction
		self.alpha = 1.0 # line-search step size
		self.gamma = 1.0 # H_0 = \gamma I
		self.sk = None
		self.yk = None
		self.termination_criterion = False
		self.min_grad = 1E-5

		# saving learning params
		self.loss_list = []
		self.grad_norm_list = []
		self.computations_time_list = []
		self.overlap_loss_list = []

		self.S = np.array([[]])
		self.Y = np.array([[]])

		self.wolfe_cond_1 = False
		self.wolfe_cond_2 = False
		self.wolfe_cond = False
		self.curvature_cond = False # this should be equal to wolfe_cond_2 when c2=1
		self.alpha_cond_1 = 1.0
		self.alpha_cond_2 = 1.0

		np.set_printoptions(precision=4)
		float_formatter = lambda x: "%.4f" % x
		np.set_printoptions(formatter={'float_kind':float_formatter})

		self.__dict__.update(kwargs) # updating input kwargs params 

	def step(self):
		start_time = time.time()
		if self.search_method == 'line-search':
			self.run_line_search_algorithm()
		elif self.search_method == 'trust-region':
			self.run_trust_region_algorithm()
		end_time = time.time()
		computation_time = end_time - start_time
		self.computations_time_list.append(computation_time)

	def run_line_search_algorithm(self):
		print('line-search iteration: ', self.k)
		self.controller.get_gk_Ok() # compute g_k^{O_k} and L_k^{O_k}
		self.controller.get_gk_Jk() # compute g_k^{J_k} and L_k^{J_k}
		self.controller.get_Lk_Jk() # compute g_k^{J_k} and L_k^{J_k}

		self.gk_Ok = self.controller.convert_gk_Ok_to_np_vec()
		self.Lk_Ok = self.controller.convert_Lk_Ok_to_np()
		self.gk = self.controller.convert_gk_Jk_to_np_vec()
		self.Lk = self.controller.convert_Lk_Jk_to_np()

		self.loss_list.append(float(self.Lk))
		self.overlap_loss_list.append(float(self.Lk_Ok))
		self.grad_norm_list.append(norm(self.gk))
		self.termination_criterion = norm(self.gk) < self.min_grad

		if self.termination_criterion:
			return

		if self.S.size == 0:
			self.pk = - self.gk # in first iteration we take the gradient decent step
		else:
			if self.search_direction_compute_method == 'two-loop' \
					   and (self.quasi_newton_matrix in ['L-BFGS','BFGS']):
				self.run_lbfgs_two_loop_recursion()

		if self.condition_method == 'Wolfe':
			self.satisfy_Wolfe_conditions()
		elif self.condition_method == 'Armijo':
			self.satisfy_Armijo_condition()
			self.controller.get_gkp1_Ok()
			self.gkp1_Ok = self.controller.convert_gkp1_Ok_to_np_vec()

		self.yk = self.gkp1_Ok - self.gk_Ok
		self.curvature_cond = (self.yk @ self.sk > 0) and not isclose(self.yk @ self.sk, 0, rel_tol=1e-05)
		if self.curvature_cond:
			print('curvature condition --> satisfy')
			print('norm(sk)      = {0:.4f}' .format(norm(self.sk)))
			print('norm(yk)      = {0:.4f}' .format(norm(self.yk)))
			print('s @ y = {0:.4f}' .format(self.yk @ self.sk))
			self.update_S_Y()
			self.gamma = (self.sk @ self.yk) / (self.yk @ self.yk)
			print('gamma before bound = {0:.4f}' .format(self.gamma))
			self.gamma = min(500.0, self.gamma) # upper bound
			self.gamma = max(1.0,   self.gamma) # lower bound
			print('gamma after  bound = {0:.4f}' .format(self.gamma))
		else:
			print('curvature condition did not satisfy -- ignoring (s,y) pair')

		if not self.wolfe_cond and self.condition_method=='Wolfe'\
						and self.ignore_step_if_wolfe_not_satisfied:
			print('ignore step since Wolfe conditions did not satisfied')
			self.controller.revert_params_to_wk()

		self.controller.update_iter_to_kp1()
		self.k += 1

	def satisfy_Wolfe_conditions(self):
		print('finding step length via running Wolfe Condition')
		print('norm(gk_Ok)   = {0:.4f}' .format(norm(self.gk_Ok)))
		print('norm(gk)      = {0:.4f}' .format(norm(self.gk)))
		print('norm(pk)      = {0:.4f}' .format(norm(self.pk)))
		print('Lk_Ok         = {0:.4f}' .format(float(self.Lk_Ok)))
		print('pk @ gk       = {0:.4f}' .format(self.pk @ self.gk))	

		self.alpha = 1.0
		rho_ls = 0.9
		c1 = 1E-4
		c2 = 0.9
		trial = 0
		first_time_cond_1 = True
		first_time_cond_2 = True
		while True: 
			self.sk = self.alpha * self.pk
			print('norm(sk)      = {0:.4f}' .format(norm(self.sk)))
			self.controller.set_sk(sk_vec=self.sk)
			self.controller.set_wkp1()
			self.controller.update_params_to_wkp1()
			self.controller.get_gkp1_Ok()
			self.gkp1_Ok = self.controller.convert_gkp1_Ok_to_np_vec()
			self.Lkp1_Ok = self.controller.convert_Lkp1_Ok_to_np()
			print('alpha = {0:.4f}' .format(self.alpha))
			print('norm(gkp1_Ok) = {0:.4f}' .format(norm(self.gkp1_Ok)))
			print('Lkp1_Ok       = {0:.4f}' .format(float(self.Lkp1_Ok)))

			lhs = self.Lkp1_Ok
			rhs = self.Lk_Ok + c1 * self.alpha * self.pk @ self.gk
			self.wolfe_cond_1 = (lhs <= rhs) # sufficient decrease

			lhs = self.gkp1_Ok @ self.pk
			rhs = c2 * self.gk @ self.pk
			self.wolfe_cond_2 = (lhs >= rhs) # curvature condition

			self.wolfe_cond = self.wolfe_cond_1 and self.wolfe_cond_2

			# record step length that satisfies Wolfe's first condition
			if self.wolfe_cond_1 and first_time_cond_1:
				self.alpha_cond_1 = self.alpha
				first_time_cond_1 = False

			# record step length that satisfies Wolfe's second condition
			if self.wolfe_cond_2 and first_time_cond_2:
				self.alpha_cond_2 = self.alpha
				first_time_cond_2 = False

			if self.wolfe_cond:
				print('Wolfe conditions --> satisfied')
				print('alpha = {0:.4f}' .format(self.alpha))
				break
			
			if self.alpha * rho_ls < 0.1:
				print('WARNING! Wolfe condition did not satisfy')
				break

			self.alpha = self.alpha * rho_ls
			trial += 1

	def satisfy_Armijo_condition(self):
		print('finding step length via running Armijo Condition')
		self.alpha = 1.0
		rho_ls = 0.9
		c1 = 1E-4
		while True: 
			self.sk = self.alpha * self.pk
			self.controller.set_sk(sk_vec=self.sk)
			self.controller.set_wkp1()
			self.controller.update_params_to_wkp1()
			self.controller.get_only_Lkp1_Ok()
			self.Lkp1_Ok = self.controller.convert_Lkp1_Ok_to_np()

			lhs = self.Lkp1_Ok
			rhs = self.Lk_Ok + c1 * self.alpha * self.pk @ self.gk
			self.wolfe_cond_1 = (lhs <= rhs) # sufficient decrease

			if self.wolfe_cond_1:
				print('Armijo condition --> satisfied')
				print('alpha = {0:.4f}' .format(self.alpha))
				break
			
			if self.alpha * rho_ls < 0.1:
				print('WARNING! Armijo condition did not satisfy')
				break

			self.alpha = self.alpha * rho_ls

	def update_S_Y(self):
		if self.S.size == 0:
			self.S = self.sk.reshape(-1,1)
			self.Y = self.yk.reshape(-1,1)
			return

		if self.S.shape[1] == self.m and (self.quasi_newton_matrix in ['L-BFGS','L-SR1']): 
			self.S = np.delete(self.S, obj=0, axis=1)
			self.Y = np.delete(self.Y, obj=0, axis=1)

		self.S = np.concatenate((self.S,self.sk.reshape(-1,1)), axis=1)
		self.Y = np.concatenate((self.Y,self.yk.reshape(-1,1)), axis=1)

	def run_lbfgs_two_loop_recursion(self):
		print('running two loop recursion')
		alpha_vec = [0]*self.S.shape[1]
		rho_vec = [0]*self.S.shape[1]
		q = self.gk
		for i in range(self.S.shape[1]-1, -1, -1):
			s = self.S[:,i]
			y = self.Y[:,i]
			rho = 1 / (y @ s)
			rho_vec[i] = rho
			alpha = rho * s @ q 
			alpha_vec[i] = alpha
			q = q - alpha * y

		r = self.gamma * q		
		for i in range(0,self.S.shape[1]):
			s = self.S[:,i]
			y = self.Y[:,i]
			beta = rho_vec[i] * y @ r
			r = r + s * (alpha_vec[i] - beta)

		self.pk = - r

	def run_trust_region_algorithm(self):
		pass
		