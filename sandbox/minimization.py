import numpy as np
from scipy.optimize import minimize

# def get_error(array):
# 	x = array[0]
# 	y = array[1]
# 	z = array[2]
# 	return x*x - 10*x + y*y + z*z + 25

# def get_grads(array):
# 	x = array[0]
# 	y = array[1]
# 	z = array[2]
# 	return np.array([2*x-10, 2*y, 2*z])

# def func(array):
# 	x = array[0]
# 	y = array[1]
# 	z = array[2]
# 	error = x*x - 10*x + y*y + z*z + 25
# 	grads = np.array([2*x-10, 2*y, 2*z])
# 	return (error, grads)

class tester:

	def __init__(self):
		self.x = 4.0
		self.y = 5.0
		self.z = 2.0
		self.c = 4.0

	def flatten(self):
		return np.array([self.x, self.y, self.z])

	def func(self, array):
		x = array[0]
		y = array[1]
		z = array[2]
		error = x*x - 2*self.c*x + y*y + z*z + self.c*self.c
		grads = np.array([2*x-2*self.c, 2*y, 2*z])
		return (error, grads)

	def update(self):
		var = self.flatten()
		res = minimize(self.func, var, method='L-BFGS-B', jac=True)
		print res.x

if __name__ == '__main__':
	t = tester()
	t.update()