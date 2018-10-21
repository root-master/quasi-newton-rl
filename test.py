import gym

env = gym.make('SpaceInvaders-v0')

s = env.reset()

score = 0.0
for t in range(100000):
	env.render()
	a = env.action_space.sample()
	sp,r,done,info = env.step(a)
	score += r
	print('reward = ',r)
	print('info = ',info)
	if done:
		print('game score = ', score)
		break