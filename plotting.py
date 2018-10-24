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

for m in memory_list: 
	for b in batch_list:
		print('L-BFGS memory = ', m)
		print('batch size = ', b)
		for task in tasks:
			file_name = 'task_' + task + '_search_line-search_matrix_L-BFGS_memory_' + str(m) + '_batch_' + str(b) + '.pkl'
			with open(file_name, 'rb') as f:
				A = pickle.load(f)
				print('max score for ',task,' = ',max(max(A[1]),max(A[4])))


for task in tasks:
	max_score = -float('Inf')
	for m in memory_list: 
		for b in batch_list:
			file_name = 'task_' + task + '_search_line-search_matrix_L-BFGS_memory_' + str(m) + '_batch_' + str(b) + '.pkl'
			with open(file_name, 'rb') as f:
				A = pickle.load(f)
				max_score = max(max_score, max(max(A[1]),max(A[4])))
	print('max score for ',task,' = ',max_score)