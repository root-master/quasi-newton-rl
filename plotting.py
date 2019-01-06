import pickle
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size': 16})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('xtick', labelsize=24)
rc('ytick', labelsize=24)
rc('axes', labelsize=30)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tasks = ['BeamRider-v0',
		 'Breakout-v0',
		 'Enduro-v0',
		 # 'Pong-v0',
		 'Qbert-v0',
		 'Seaquest-v0',
		 'SpaceInvaders-v0']

batch_list = [512,1024,2048,4096]
memory_list = [20,40,80]
results_folder = './results/'
plots_folder = './plots/'
times = {}
max_score_simulations = {}
for task in tasks:
	times[task] = []
	max_score_simulations[task] = []
for m in memory_list: 
	for b in batch_list:
		print('-'*60)
		print('L-BFGS memory = ', m)
		print('batch size = ', b)
		for task in tasks:			
			file_name = 'task_' + task + '_search_line-search_matrix_L-BFGS_memory_' + str(m) + '_batch_' + str(b) + '.pkl'
			file_path = results_folder + file_name
			with open(file_path, 'rb') as f:
				A = pickle.load(f)
				print('max score for ',task,' = ',max(max(A[1]),max(A[4])))
				hour = int(A[5]/3600.0)
				minute = int( (A[5] - 3600 * hour)/60.0)
				print('time = ', hour, 'hours and ',minute, 'minutes')
				times[task].append(hour+minute/60.0)
				max_score_simulations[task].append(max(max(A[1]),max(A[4])))

mean_time = []
std_time = []
for task in tasks:
	mean_time.append(np.mean(times[task],axis=0))
	std_time.append(np.std(times[task],axis=0))
x_vec = list(task[:-3] for task in tasks  )

rc('xtick', labelsize=12)

plt.figure(figsize=(8, 4))
plt.xlabel('Tasks')
plt.ylabel('Train time (hours)')
plt.errorbar(x_vec, mean_time,std_time,linestyle='None', marker='s',color='b')
plot_path = plots_folder + 'train_time' + '.pdf'
plt.savefig(plot_path, format='pdf', dpi=1000,bbox_inches='tight',pad_inches = 0)

mean_scores = []
std_scores = []
for task in tasks:
	mean_scores.append(np.mean(max_score_simulations[task],axis=0))
	std_scores.append(np.std(max_score_simulations[task],axis=0))
x_vec = list(task[:-3] for task in tasks  )

rc('xtick', labelsize=12)

plt.figure(figsize=(8, 4))
plt.xlabel('Tasks')
plt.ylabel('Game Scores')
plt.errorbar(x_vec, mean_scores,std_scores,linestyle='None', marker='s',color='b')
plot_path = plots_folder + 'game_scores' + '.pdf'
plt.savefig(plot_path, format='pdf', dpi=1000,bbox_inches='tight',pad_inches = 0)


rc('xtick', labelsize=24)
rc('ytick', labelsize=24)

best_result = {}
print('-'*60)
for task in tasks:
	max_score = -float('Inf')
	for m in memory_list: 
		for b in batch_list:
			file_name = 'task_' + task + '_search_line-search_matrix_L-BFGS_memory_' + str(m) + '_batch_' + str(b) + '.pkl'
			file_path = results_folder + file_name
			with open(file_path, 'rb') as f:
				A = pickle.load(f)
				if max(max(A[1]),max(A[4])) >= max_score:
					max_score = max(max_score, max(max(A[1]),max(A[4])))
					hour = int(A[5]/3600.0)
					minute = int( (A[5] - 3600 * hour)/60.0)
					m_best = m
					b_best = b
					best_result[task] = A

	print('max score for ',task,' = ',max_score, '--','b = ',b_best,' -- m = ',m_best)
	print('time = ', hour, 'hours and ',minute, 'minutes')


# for task in tasks:
# 	results = best_result[task]
# 	train_scores = results[1]
# 	stop_step = results[8]
# 	num_intervals = len(train_scores) // 20
# 	interval_step = stop_step // num_intervals
# 	x_vec = []
# 	max_train_scores = []
# 	for i in range(0,num_intervals):
# 		x_vec.append(interval_step*i)
# 		max_train_scores.append(max(train_scores[20*i:20*(i+1)]))
# 	legend = 'train scores - ' + task[:-3]
# 	plt.figure()
# 	plt.xlabel('Episode Steps')
# 	plt.ylabel('Training Scores')
# 	plt.plot(x_vec, max_train_scores,'.-',label=legend,markersize=0.5,linewidth=0.5)
# 	plt.legend(loc=1)

## THIS IS GOOD -- LOSS ##
for task in tasks:
	results = best_result[task]
	train_loss = results[9]
	if task == 'BeamRider-v0':
		train_loss.reverse()
	if task == 'Seaquest-v0':
		train_loss.reverse()
	if task == 'Qbert-v0':
		train_loss.reverse()
	if task == 'SpaceInvaders-v0':
		train_loss = train_loss[:315]

	stop_step = results[8]
	num_intervals = len(train_loss) // 2
	interval_step = stop_step // num_intervals
	x_vec = []
	min_train_loss = []
	for i in range(0,num_intervals):
		x_vec.append(interval_step*i)
		min_train_loss.append(min(train_loss[2*i:2*(i+1)]))

	legend = 'Train loss - ' + task[:-3]
	plt.figure(figsize=(8, 4))
	plt.xlabel('Episode Steps')
	plt.ylabel('Train Loss')
	plt.locator_params(axis='x', nbins=5)
	plt.plot(x_vec, min_train_loss,'.-',label=legend,markersize=0.5,linewidth=0.5,color='r')
	plt.legend(loc=1)
	plot_path = plots_folder + 'train_loss_' + task[:-3] + '.pdf'
	plt.savefig(plot_path, format='pdf', dpi=1000,bbox_inches='tight',pad_inches = 0)

# ## THIS IS GOOD -- TEST SCORES ##
for task in tasks:
	results = best_result[task]
	test_scores = results[4]
	stop_step = results[8]
	num_intervals = len(test_scores) // 5
	interval_step = stop_step // num_intervals
	x_vec = []
	max_test_scores = []
	for i in range(0,num_intervals):
		x_vec.append(interval_step*i)
		max_test_scores.append(max(test_scores[5*i:5*(i+1)]))
	legend = 'Test scores -- ' + task[:-3]
	plt.figure(figsize=(8, 4))
	plt.xlabel('Episode Steps')
	plt.ylabel('Game Scores')
	plt.locator_params(axis='x', nbins=5)
	plt.plot(x_vec, max_test_scores,'.-',label=legend,markersize=0.5,linewidth=0.5,color='b')
	plt.legend(loc=1)
	plot_path = plots_folder + 'test_scores_' + task[:-3] + '.pdf'
	plt.savefig(plot_path, format='pdf', dpi=1000,bbox_inches='tight',pad_inches = 0)

# for task in tasks:
# 	results = best_result[task]
# 	test_scores = results[1]
# 	stop_step = results[8]
# 	num_intervals = len(test_scores) // 5
# 	interval_step = stop_step // num_intervals
# 	x_vec = []
# 	max_test_scores = []
# 	for i in range(0,num_intervals):
# 		x_vec.append(interval_step*i)
# 		max_test_scores.append(max(test_scores[5*i:5*(i+1)]))
# 	legend = 'test scores - ' + task[:-3]
# 	plt.figure()
# 	plt.xlabel('Episode Steps')
# 	plt.ylabel('Test Scores')
# 	plt.locator_params(axis='x', nbins=5)
# 	plt.plot(x_vec, max_test_scores,'.-',label=legend,markersize=0.5,linewidth=0.5,color='b')
# 	plt.legend(loc=1)

# 0:	episode_steps_list,
# 1:	episode_scores_list,
# 2:	episode_rewards_list,
# 3:	episode_time_list,
# 4:	testing_scores,
# 5:	total_train_time,
# 6:	batch_size,
# 7:	seed,
# 8:	step,
# 9:	quasi_newton.loss_list,
# 10:	quasi_newton.grad_norm_list,
# 11:	quasi_newton.computations_time_list


# task = 'BeamRider-v0'
# results = best_result[task]
# train_loss = results[9][:-2]
# x_vec = range(len(train_loss))
# legend = task[:-3]
# plt.figure()
# plt.plot(x_vec, train_loss,'-',label=legend,markersize=10)
# plt.legend(loc=1)
# plt.show()

# task = 'Breakout-v0'
# results = best_result[task]
# train_loss = results[9][:-2]
# x_vec = x_vec = range(len(train_loss))
# legend = task[:-3]
# plt.figure()
# plt.plot(x_vec, train_loss,'-',label=legend,markersize=10)
# plt.legend(loc=1)
# plt.show()

# batch_list = [8192,4096]
# memory_list = [160]
# results_folder = './results/'
# best_result = {}

# for m in memory_list: 
# 	for b in batch_list:
# 		print('-'*60)
# 		print('L-BFGS memory = ', m)
# 		print('batch size = ', b)
# 		for task in tasks:
# 			file_name = 'task_' + task + '_search_line-search_matrix_L-BFGS_memory_' + str(m) + '_batch_' + str(b) + '.pkl'
# 			file_path = results_folder + file_name
# 			with open(file_path, 'rb') as f:
# 				A = pickle.load(f)
# 				best_result[task] = A
# 				print('max score for ',task,' = ',max(max(A[1]),max(A[4])))
# 				hour = int(A[5]/3600.0)
# 				minute = int( (A[5] - 3600 * hour)/60.0)
# 				print('time = ', hour, 'hours and ',minute, 'minutes')


# for task in tasks:
# 	results = best_result[task]
# 	scores = results[4]
# 	x_vec = range(len(scores))
# 	legend = 'episode score ' + task[:-3]
# 	plt.figure()
# 	plt.plot(x_vec, scores,'-',label=legend,markersize=1,linewidth=0.5)
# 	plt.legend(loc=1)


# plt.show()


# import pickle
# file_name = 'task_Pong-v0_search_line-search_matrix_L-BFGS_memory_160_batch_4096.pkl'
# with open(file_name,'rb') as f:
# 	A = pickle.load(f)
# print(max(max(A[1]),max(A[4])))