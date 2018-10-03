from math import isclose, sqrt
import numpy as np
from numpy.linalg import inv, qr, norm, pinv
from scipy.linalg import eig, eigvals

class LBFGS():
	"""docstring for L_BFGS"""
	def __init__(self,
				controller=None,
				m=20,
				**kwargs):
		self.controller = controller
		self.m = m
		self.k = 0 # l-bfgs counter
		self.gk = None # current gradient on J_k
		self.gk_Ok = None # current gradient on O_k
		self.gkp1_Ok = None # next gradient on O_k
		self.Lk = 0.0 # current loss
		self.Lkp1 = 0.0 # next loss 
		self.pk = None # line-search step direction
		self.alpha = 1.0 # line-search step size
		self.gamma = 1.0 # H_0 = \gamma I
		self.sk = None
		self.yk = None

		self.S = np.array([[]])
		self.Y = np.array([[]])

		self.wolfe_cond_1 = False
		self.wolfe_cond_2 = False
		self.wolfe_cond = False
		self.curvature_cond = False # this should be equal to wolfe_cond_2 when c2=1
		self.alpha_cond_1 = 1.0
		self.alpha_cond_2 = 1.0

		self.__dict__.update(kwargs) # updating input kwargs params 

	def run_line_search_algorithm(self):
		print('line-search iteration: ', self.k)
		self.controller.get_gk_Ok() # compute g_k^{O_k} and L_k^{O_k}
		self.controller.get_gk_Jk() # compute g_k^{J_k} and L_k^{J_k}
		self.gk = self.controller.convert_gk_Jk_to_np_vec()

		if self.S.size == 0:
			self.pk = - self.gk # in first iteration we take the gradient decent step
		else:
			self.run_lbfgs_two_loop_recursion()

		self.satisfy_Wolfe_conditions()

		self.yk = self.gkp1_Ok - self.gk_Ok
		self.curvature_cond = (self.yk @ self.sk > 0) and not isclose(self.yk @ self.sk, 0)
		if self.curvature_cond:
			print('curvature condition --> satisfy')
			self.update_S_Y()
			self.gamma = (self.sk @ self.yk) / (self.yk @ self.yk)
			self.gamma = min(500.0, self.gamma) # upper bound
			self.gamma = max(1.0,   self.gamma) # lower bound
		else:
			print('curvature condition did not satisfy -- ignoring (s,y) pair')
			self.gamma = 1.0

		self.controller.update_iter_to_kp1()
		self.k += 1

	def satisfy_Wolfe_conditions(self):
		self.gk_Ok = self.controller.convert_gk_Ok_to_np_vec()
		self.Lk_Ok = self.controller.convert_Lk_Ok_to_np()
		self.alpha = 1.0
		rho_ls = 0.9
		c1 = 1E-4
		c2 = 1.0
		trial = 0
		first_time_cond_1 = True
		first_time_cond_2 = True
		while True: 
			self.sk = self.alpha * self.pk
			self.controller.set_sk(sk_vec=self.sk)
			self.controller.set_wkp1()
			self.controller.update_params_to_wkp1()
			self.controller.get_gkp1_Ok()
			self.gkp1_Ok = self.controller.convert_gkp1_Ok_to_np_vec()
			self.Lkp1_Ok = self.controller.convert_Lkp1_Ok_to_np()

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

			if self.wolfe_cond_1 and self.wolfe_cond_2:
				print('Wolfe conditions --> satisfied')
				print('alpha = {0:.4f}' .format(self.alpha))
				break
			
			if self.alpha * rho_ls < 0.1:
				print('WARNING! Wolfe condition did not satisfy')
				break

			self.alpha = self.alpha * rho_ls
			trial += 1

	def update_S_Y(self):
		if self.S.size == 0:
			self.S = self.sk.reshape(-1,1)
			self.Y = self.yk.reshape(-1,1)
			return

		if self.S.shape[1] == self.m: 
			self.S = np.delete(self.S, obj=0, axis=1)
			self.Y = np.delete(self.Y, obj=0, axis=1)

		self.S = np.concatenate((self.S,self.sk.reshape(-1,1)), axis=1)
		self.Y = np.concatenate((self.Y,self.yk.reshape(-1,1)), axis=1)

	def run_lbfgs_two_loop_recursion(self):
		print('running two loop recursion')
		alpha_vec = []
		rho_vec = []
		q = self.gk
		for i in range(self.S.shape[1]-1, -1, -1):
			s = self.S[:,i]
			y = self.Y[:,i]
			rho = 1 / (y @ s)
			rho_vec.append(rho)
			alpha = rho * s @ q 
			alpha_vec.append(alpha)
			q = q - alpha * y

		r = self.gamma * q		
		for i in range(0,self.S.shape[1]):
			s = self.S[:,i]
			y = self.Y[:,i]
			beta = rho_vec[i] * y @ r
			r = r + s * (alpha_vec[i] - beta)

		self.pk = - r
		