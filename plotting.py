import pickle
tasks = ['BeamRider-v0',
		 'Breakout-v0',
		 'Enduro-v0',
		 'Pong-v0',
		 'Qbert-v0',
		 'Seaquest-v0',
		 'SpaceInvaders-v0']

batch_list = [512,1024,2048,4096]
memory_list = [20,40,80]
results_folder = './results/'
for m in memory_list: 
	for b in batch_list:
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


for task in tasks:
	max_score = -float('Inf')
	for m in memory_list: 
		for b in batch_list:
			file_name = 'task_' + task + '_search_line-search_matrix_L-BFGS_memory_' + str(m) + '_batch_' + str(b) + '.pkl'
			file_path = results_folder + file_name
			with open(file_path, 'rb') as f:
				A = pickle.load(f)
				if max(max(A[1]),max(A[4])) > max_score:
					max_score = max(max_score, max(max(A[1]),max(A[4])))
					hour = int(A[5]/3600.0)
					minute = int( (A[5] - 3600 * hour)/60.0)

	print('max score for ',task,' = ',max_score)
	print('time = ', hour, 'hours and ',minute, 'minutes')

batch_list = [8192]
memory_list = [160]
results_folder = './results/'
for m in memory_list: 
	for b in batch_list:
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


# import pickle
# file_name = 'task_Pong-v0_search_line-search_matrix_L-BFGS_memory_160_batch_4096.pkl'
# with open(file_name,'rb') as f:
# 	A = pickle.load(f)
# print(max(max(A[1]),max(A[4])))