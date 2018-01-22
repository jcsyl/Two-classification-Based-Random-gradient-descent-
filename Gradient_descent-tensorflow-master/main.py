import numpy as np
import Network
def vectorized_result(j):
	e = np.zeros((2,1))
	e[j]=1.0
	return e
###########################################################################
#		2-2-1 三层神经网络偏置和权值设置如下
#		input-hidden biases:[0.5,-0.5]  hidden-output biases:[1]
#  		input-hidden weights:[[0.3,-0.4],[-0.1,1.0]]  hidden-output:[-2.0,0.5]
###########################################################################
#		data1_input 数据用于第二问的梯度下降分类的训练
#		data2_input 数据用于正向传播得到输出的正负号
###########################################################################
data1_input = [[0.28,1.31,-6.2],[0.07,0.58,-0.78],[1.54,2.01,-1.63],[-0.44,1.18,-4.32],[-0.81,0.21,5.73],
				[1.52,3.16,2.77],[2.20,2.42,-0.19],[0.91,1.94,6.21],[0.65,1.93,4.38],[-0.26,0.82,-0.96],
				[0.011,1.03,-0.21],[1.27,1.28,0.08],[0.13,3.12,0.16],[0.21,1.23,-0.11],[-2.18,1.39,-0.19],
				[0.34,1.96,-0.16],[-1.38,0.94,0.45],[-0.12,0.82,0.17],[-1.44,2.31,0.14],[0.26,1.94,0.08]]
data1_lable = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1]		

data2_input = [[0.28,1.31],[0.07,0.58],[1.54,2.01],[-0.44,1.18],[-0.81,0.21],
				[1.52,3.16],[2.20,2.42],[0.91,1.94],[0.65,1.93],[-0.26,0.82],
				[0.011,1.03],[1.27,1.28],[0.13,3.12],[-0.21,1.23],[-2.18,1.39],
				[0.34,1.96],[-1.38,0.94],[-0.12,0.82],[-1.44,2.31],[0.26,1.94],
				[1.36,2.17],[1.41,1.45],[1.22,0.99],[2.46,2.19],[0.68,0.79],
				[2.51,3.22],[0.60,2.44],[0.64,0.13],[0.85,0.58],[0.66,0.51]]	

############################################################
#		1.a
def one_a():
	biaes = [[0.5,-0.5],[1]]
	weight1 = np.array([[0.3,-0.4],[-0.1,1.0]])
	weight2 = np.array([-2.0,0.5])
	weights = []
	weights.append(weight1)
	weights.append(weight2)
	train_data = [np.reshape(x,(2,1)) for x in data2_input]
	net = Network.Network([2,2,1],biaes,weights)
	result  = net.get_sign(train_data)
	print(result)
############################################################
#		1.b
def one_b():
	biaes = [[-1.0,1.0],[0.5]]
	weight1 = np.array([[-0.5,1.5],[1.5,-0.5]])
	weight2 = np.array([-1.0,1.0])
	weights = []
	weights.append(weight1)
	weights.append(weight2)
	train_data = [np.reshape(x,(2,1)) for x in data2_input]
	net = Network.Network([2,2,1],biaes,weights)
	result  = net.get_sign(train_data)
	print(result)

############################################################
#		2.a
def Two_a():
	train_input = [np.reshape(x,(3,1)) for x in data1_input]
	#train_lable = [vectorized_result(y) for y in data1_lable]
	train_data = zip(train_input,data1_lable)
	net = Network.Network([3,1,1])
	net.SGD(train_data,1000,0.5)
############################################################
#		2.b
def Two_b():
	biaes = [[0.5],[-0.5]]
	weight1 = np.array([0.5,0.5,0.5])
	weight2 = np.array([-0.5])
	weights = []
	weights.append(weight1)
	weights.append(weight2)
	train_input= [np.reshape(x,(3,1)) for x in data1_input]
	train_data = zip(train_input,data1_lable)
	net = Network.Network([3,1,1],biaes,weights)
	net.SGD(train_data,1000,0.5)

if __name__=='__main__':
	# one_a()
	# one_b()
	#Two_a()
	print("#######################################")
	Two_b()
