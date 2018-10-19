import pickle

file_name = 'task_Breakout-v0_search_line-search_matrix_L-BFGS_memory_40_batch_2048.pkl'
with open(file_name, 'rb') as f:
	A = pickle.load(f)