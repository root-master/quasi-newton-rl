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

		# trust region parameters
		self.delta_hat = 0.5 # upper bound for trust region radius
		self.delta_vec = [] # trust region radii
		self.delta = self.delta_hat * 0.75 # current trust region radius
		# self.delta_vec.append(self.delta)
		# self.rho_vec = [] # true reduction / predicted reduction ratio
		# self.eta = 1/4 * 0.9 # eta \in [0,1/4)

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
		self.curvature_cond = (self.yk @ self.sk > 0) and \
				not isclose(self.yk @ self.sk, 0, rel_tol=1e-05)
		if self.curvature_cond:
			print('curvature condition --> satisfy')
			print('norm(sk)      = {0:.4f}' .format(norm(self.sk)))
			print('norm(yk)      = {0:.4f}' .format(norm(self.yk)))
			print('s @ y = {0:.4f}' .format(self.yk @ self.sk))
			self.update_S_Y()
			self.gamma = (self.sk @ self.yk) / (self.yk @ self.yk)
			print('gamma before bound = {0:.4f}' .format(self.gamma))
			self.gamma = min(1000.0, self.gamma) # upper bound
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
		return

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
		return

	def update_S_Y(self):
		if self.S.size == 0:
			self.S = self.sk.reshape(-1,1)
			self.Y = self.yk.reshape(-1,1)
			return

		if self.S.shape[1] == self.m and \
			(self.quasi_newton_matrix in ['L-BFGS','L-SR1']): 
			self.S = np.delete(self.S, obj=0, axis=1)
			self.Y = np.delete(self.Y, obj=0, axis=1)

		self.S = np.concatenate((self.S,self.sk.reshape(-1,1)), axis=1)
		self.Y = np.concatenate((self.Y,self.yk.reshape(-1,1)), axis=1)
		return

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
		eta = 0.99 * 1e-3
		print('trust region iteration: ', self.k)
		self.k += 1
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
			print('gradient vanished')
			print('convergence necessary but not sufficient condition') 
			print('--BREAK -- the trust region loop!')
			return

		if self.S.size == 0:
			# in first iteration we take the gradient decent step
			self.pk = - self.gk
			self.satisfy_Wolfe_conditions()
		else:
			self.find_pk_trust_region()

		self.controller.get_gkp1_Ok()
		self.gkp1_Ok = self.controller.convert_gkp1_Ok_to_np_vec()
		self.yk = self.gkp1_Ok - self.gk_Ok

		# k = 0
		if self.S.size == 0:
			print('norm(sk)      = {0:.4f}' .format(norm(self.sk)))
			print('norm(yk)      = {0:.4f}' .format(norm(self.yk)))
			print('s @ y = {0:.4f}' .format(self.yk @ self.sk))
			self.update_S_Y()
			self.gamma = (self.sk @ self.yk) / (self.yk @ self.yk)
			print('gamma before bound = {0:.4f}' .format(self.gamma))
			self.gamma = min(1000.0, self.gamma) # upper bound
			self.gamma = max(1.0,   self.gamma) # lower bound
			print('gamma after bound = {0:.4f}' .format(self.gamma))
			self.controller.update_iter_to_kp1()
			return

		self.sk = self.pk
		self.controller.set_sk(sk_vec=self.sk)
		self.controller.set_wkp1()
		self.controller.update_params_to_wkp1()
		self.controller.get_only_Lkp1_Ok()
		self.Lkp1_Ok = self.controller.convert_Lkp1_Ok_to_np()

		self.Lkp1_Ok = self.controller.convert_Lkp1_Ok_to_np()
		print('Lkp1_Ok       = {0:.4f}' .format(float(self.Lkp1_Ok)))
		ared = self.Lk_Ok - self.Lkp1_Ok
		print('ared       = {0:.4f}' .format(float(ared)))

		p_ll = self.P_ll.T @ self.pk
		p_NL_norm = sqrt ( abs( norm(self.pk) ** 2 - norm(p_ll) ** 2 ) )
		p_T_B_p = sum( self.Lambda_1 * p_ll ** 2)  + self.gamma * p_NL_norm ** 2
		pred =  - (self.gk @ self.pk  + 1/2 * p_T_B_p)
		print('pred       = {0:.4f}' .format(float(pred)))

		rho = ared / pred

		self.rho = rho
		self.ared = ared
		self.pred = pred

		if rho > eta:
			if norm(self.sk) <= 0.8 * self.delta:
				self.delta = self.delta
			else:
				self.delta = 2 * self.delta
		elif 0.1 <= rho and rho <= 0.75:
			self.delta = self.delta
		else:
			self.delta = 0.5 * self.delta 

		curvature_cond = True
		if self.quasi_newton_matrix in ['L-BFGS','BFGS']:
			if (self.yk @ self.sk < 0) or \
				isclose(self.yk @ self.sk,0.0, rel_tol=1e-05):
				curvature_cond = False
			else:
				curvature_cond = True

		# something similar to Eq (6.26) of book		
		lhs = 0.0
		if self.quasi_newton_matrix in ['L-SR1','SR1']:
			s_ll = self.P_ll.T @ self.sk
			s_NL_norm = sqrt ( abs( norm(self.sk) ** 2 - norm(s_ll) ** 2 ) )
			sTBs = sum( self.Lambda_1 * s_ll ** 2) + self.gamma * s_NL_norm ** 2
			lhs = abs(self.sk @ self.yk - sTBs)
			if isclose(lhs,0.0,rel_tol=1e-06) or \
				 isclose(norm(self.sk),0.0,rel_tol=1e-06) or \
				 		isclose(norm(self.yk),0.0,rel_tol=1e-06):
				curvature_cond = False
			else:
				curvature_cond = True

		if self.curvature_cond:
			print('curvature condition --> satisfy')
			print('norm(sk)      = {0:.4f}' .format(norm(self.sk)))
			print('norm(yk)      = {0:.4f}' .format(norm(self.yk)))
			print('s @ y = {0:.4f}' .format(self.yk @ self.sk))
			self.update_S_Y()
			self.gamma = (self.sk @ self.yk) / (self.yk @ self.yk)
			print('gamma before bound = {0:.4f}' .format(self.gamma))
			self.gamma = min(1000.0, self.gamma) # upper bound
			self.gamma = max(1.0,   self.gamma) # lower bound
			print('gamma after bound = {0:.4f}' .format(self.gamma))
		else:
			print('curvature condition did not satisfy -- ignoring (s,y) pair')

		if rho > eta:
			self.controller.update_iter_to_kp1()
		else:
			print('ignore step ared/pred < eta')
			self.controller.revert_params_to_wk()


	def find_pk_trust_region(self):
		if self.quasi_newton_matrix in ['L-BFGS','BFGS']:
			self.lbfgs_trust_region_subproblem_solver()
		if self.quasi_newton_matrix in ['L-R1','SR1']:
			self.lsr1_trust_region_subproblem_solver()

	def lbfgs_trust_region_subproblem_solver(self):
		n = self.pk.size
		S = self.S
		Y = self.Y
		g = self.gk
		gamma = self.gamma
		delta = self.delta

		Psi = np.concatenate( (gamma*S, Y) ,axis=1)
		
		S_T_Y = S.T @ Y
		L = np.tril(S_T_Y,k=-1)
		U = np.tril(S_T_Y.T,k=-1).T
		D = np.diag( np.diag(S_T_Y) )

		M = - inv( np.block([ 	[gamma * S.T @ S ,	L],
								[     L.T,		   -D] 
				]) )

		M = (M + M.T) / 2 

		Q, R = qr(Psi, mode='reduced')
		eigen_values, eigen_vectors = eig( R @ M @ R.T )
		eigen_values = eigen_values.real

		# sorted eigen values
		idx = eigen_values.argsort()
		eigen_values_sorted = eigen_values[idx]
		eigen_vectors_sorted = eigen_vectors[:,idx]

		Lambda_hat = eigen_values_sorted
		V = eigen_vectors_sorted

		Lambda_1 = gamma + Lambda_hat
		# Lambda_2 = gamma * np.ones( n-len(Lambda_hat) )
		# B_diag = np.concatenate( (Lambda_1, Lambda_2),axis=0 )


		P_ll = Psi @ inv(R) @ V # P_parallel 
		g_ll = P_ll.T @ g	# g_Parallel
		g_NL_norm = sqrt ( abs( norm(g) ** 2 - norm(g_ll) ** 2 ) )

		self.P_ll = P_ll
		self.g_ll = g_ll
		self.g_NL_norm = g_NL_norm
		self.Lambda_1 = Lambda_1

		sigma = 0
		phi = self.phi_bar_func(sigma,delta)

		if phi >= 0:
			sigma_star = 0
			tau_star = gamma
		else:
			sigma_star = self.solve_newton_equation_to_find_sigma()
			tau_star = gamma + sigma_star

		self.pk = - 1 / tau_star * \
			( g - Psi @ inv( tau_star * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g) )


	def solve_newton_equation_to_find_sigma(self):
		# tolerance
		tol = 1E-2
		Lambda_1 = self.Lambda_1
		gamma = self.gamma
		delta = self.delta
		g_NL_norm = self.g_NL_norm
		g_ll = self.g_ll
		gamma = self.gamma

		lambda_min = min( Lambda_1.min(), gamma )
		sigma = max( 0, -lambda_min )
		counter = 0
		if self.phi_bar_func(sigma,delta) < 0:
			sigma_hat = np.max( abs( g_ll ) / delta - Lambda_1 )
			sigma_hat = max(sigma_hat , (g_NL_norm / delta - gamma) ) 
			sigma = max( 0, sigma_hat)
			while( abs( self.phi_bar_func(sigma,delta) ) > tol ):
				phi_bar = self.phi_bar_func(sigma,delta)
				phi_bar_prime = self.phi_bar_prime_func(sigma)
				sigma = sigma - phi_bar / phi_bar_prime
				counter += 1
				if counter > 1000:
					print('had to break newton solver')
					break

			sigma_star = sigma
		elif lambda_min < 0:
			sigma_star = - lambda_min
		else:
			sigma_star = 0

		return sigma_star

	def phi_bar_func(self,sigma,delta):
		# phi(sigma) = 1 / v(sigma) - 1 / delta	
		Lambda_1 = self.Lambda_1
		gamma = self.gamma
		g_ll = self.g_ll
		g_NL_norm = self.g_NL_norm

		u = sum( (g_ll ** 2) / ((Lambda_1 + sigma) ** 2) ) + \
								(g_NL_norm ** 2) / ( (gamma + sigma) ** 2)
		v = sqrt(u) 

		phi = 1 / v - 1 / delta
		return phi

	def phi_bar_prime_func(self,sigma):
		Lambda_1 = self.Lambda_1
		gamma = self.gamma
		g_ll = self.g_ll
		g_NL_norm = self.g_NL_norm

		u = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 2 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 2

		u_prime = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 3 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 3
		phi_bar_prime = u ** (-3/2) * u_prime

		return phi_bar_prime

	def lsr1_trust_region_subproblem_solver(self):
		delta = self.delta
		g = self.gk
		self.pk = -g # just for intialization
		n = g.size
		S = self.S
		Y = self.Y		
		gamma = self.gamma

		S_T_Y = S.T @ Y
		L = np.tril(S_T_Y,k=-1)
		U = np.tril(S_T_Y.T,k=-1).T
		D = np.diag( np.diag(S_T_Y) )
		M = inv(D + L + L.T - gamma * S.T @ S )
		# make sure M is symmetric
		M = (M + M.T) / 2
		Psi = Y - gamma * S
		Q, R = qr(Psi, mode='reduced')

		# check if Psi is full rank or not
		if np.isclose(np.diag(R),0).any():
			rank_deficieny = True
			# find zeros of diagonal of R
			rank_deficient_idx = np.where( np.isclose(np.diag(R),0))[0]
			# deleting the rows of R with a 0 diagonal entry (r * k+1)
			R_cross = np.delete( R, obj=rank_deficient_idx, axis=0 )
			# deleting the columsn of Psi with a 0 diagonal entry on R (n * r)
			Psi_cross = np.delete( Psi, obj=rank_deficient_idx, axis=1 )
			# deleting the rows and columns of R with a 0 diagonal entry (r * r)
			R_cross_cross = np.delete( R_cross, obj=rank_deficient_idx, axis=1 )
			# (n * r)
			Q_hat = Psi_cross @ inv(R_cross_cross)
			# (r * r)
			R_M_R_T = R_cross @ M @ R_cross.T
		else:
			rank_deficieny = False
			R_M_R_T = R @ M @ R.T

		eigen_values, eigen_vectors = eig( R_M_R_T )
		# make sure eigen values are real
		eigen_values = eigen_values.real
		eigen_vectors = eigen_vectors.real

		# sorted eigen values
		idx = eigen_values.argsort()
		eigen_values_sorted = eigen_values[idx]
		eigen_vectors_sorted = eigen_vectors[:,idx]

		Lambda_hat = eigen_values_sorted
		V = eigen_vectors_sorted

		Lambda_1 = gamma + Lambda_hat

		lambda_min = min( Lambda_1.min(), gamma )

		if rank_deficieny:
			P_ll = Psi_cross @ inv(R_cross_cross) @ V
		else:
			P_ll = Psi @ inv(R) @ V # P_parallel 
		g_ll = P_ll.T @ g	# g_Parallel
		g_NL_norm = sqrt ( abs( norm(g) ** 2 - norm(g_ll) ** 2 ) )

		self.P_ll = P_ll
		self.g_ll = g_ll
		self.g_NL_norm = g_NL_norm
		self.Lambda_1 = Lambda_1
		self.lambda_min = lambda_min

		sigma = 0

		if lambda_min > 0 and self.phi_bar_func(0,delta) >= 0:
			sigma_star = 0
			tau_star = gamma
			# Equation (11) of SR1 paper
			p_star = - 1 / tau_star * \
				( g - Psi @ inv( tau_star * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g) )
		elif lambda_min <= 0 and self.phi_bar_func(-lambda_min, delta) >= 0:
			sigma_star = -lambda_min
			if rank_deficieny:
				if ~isclose(sigma_star, -gamma):
					p_star = - Psi_cross @ inv(R_cross_cross) @ U * pinv( np.diag(Lambda_1 + sigma_star) ) @ g_ll - 1 / ( gamma + sigma_star) * ( g - ( Psi_cross @ inv(R_cross_cross) ) @ inv(R_cross_cross).T @ ( Psi_cross.T @ g ) )
				else:
					p_star = - Psi_cross @ inv(R_cross_cross) @ U * inv( np.diag(Lambda_1 + sigma_star) ) @ g_ll

			else:
				# Equation (13) of SR1 paper
				if ~isclose(sigma_star, -gamma):
					p_star = - Psi @ inv(R) @ U * pinv( np.diag(Lambda_1 + sigma_star) ) @ g_ll - 1 / ( gamma + sigma_star) * ( g - ( Psi @ inv(R) ) @ inv(R).T @ ( Psi.T @ g ) )
				else:
					p_star = - Psi @ inv(R) @ U * inv( np.diag(Lambda_1 + sigma_star) ) @ g_ll

			# so-called hard-case
			if lambda_min < 0:
				p_star_hat = p_star.copy()
				# Equation (14) of SR1 paper
				# check if lambda_min is Lambda_1[0]
				if isclose( lambda_min, Lambda_1.min()):
					u_min = P_ll[:,0].reshape(-1,1)
				else:
					for j in range(Lambda_1.size+2):
						e = np.zeros((n,1))
						e[j,0] = 1.0
						u_min = e - P_ll @ P_ll.T @ e
						if ~isclose( norm(u_min), 0.0):
							break
				# find alpha in Eq (14)
				# solve a * alpha^2 + b * alpha + c = 0  
				a = norm(u_min) ** 2
				b = 2 * norm(u_min) * norm(p_star_hat)
				c = norm(p_star_hat) - delta ** 2
				
				alpha_1 = -b + sqrt(b ** 2 - 4 * a * c) / (2 * a)
				alpha_2 = -b - sqrt(b ** 2 - 4 * a * c) / (2 * a)
				alpha = alpha_1
				
				p_star = p_star_hat + alpha * u_min 
		else:
			sigma_star = self.solve_newton_equation_to_find_sigma(delta)
			tau_star = sigma_star + gamma
			# Equation (11) of SR1 paper
			p_star = - 1 / tau_star * \
				( g - Psi @ inv(tau_star * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g))

		self.pk = p_star




		