import numpy as np

class ReplayBuffer():
	def __init__(self, max_size):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def push(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size = batch_size)
		x, u, y = [], [], []
		for i in ind:
			X, U, Y = self.storage[i]
			x.append(np.array(X, copy = False))
			y.append(np.array(Y, copy = False))
			u.append(np.array(U, copy = False))
		return np.array(x).reshape(batch_size, -1), np.array(u).reshape(batch_size, -1), np.array(y).reshape(batch_size, -1)
