""" 
2.一个实现了随机梯度下降学习算法的前馈神经网络模块。 
3.梯度的计算使用到了反向传播。 
4.由于我专注于使代码简介、易读且易于修改，所以它不是最优化的，省略了许多令人满意的特性。 
5."""  
  
#### Libraries  
#标准库  
import random  
  
#第三方库  
import numpy as np   
class Network(object):
    def __init__(self, sizes,biasess=None,weightss=None):
        if biasess:
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.biases =[np.vstack(biase) for biase in biasess]
            self.weights =weightss
        else:
            self.num_layers = len(sizes)  
            #num_layers为层数  
            self.sizes = sizes  
            #列表 sizes 包含各层神经元的数量  
            self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
            self.weights = [np.random.rand(y, x)  
                            for x, y in zip(sizes[:-1], sizes[1:])]  

    def feedforward(self, a):  
        """如果a是输入，则返回网络的输出"""  
        for b, w in zip(self.biases, self.weights):  
            a = sigmoid(np.dot(w, a)+b)  
        if a>=0:
            return 1
        else :
            return 0 
    def feedforward2(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        if a>=0:
            return 1
        else :
            return -1
 
    def SGD(self, training_data, epochs, eta):  
        """ 
        :param training_data: training_data 是一个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。 
        :param epochs: 变量 epochs为迭代期数量 
        :param mini_batch_size: 变量mini_batch_size为采样时的⼩批量数据的⼤⼩ 
        :param eta: 学习速率 
        :return: 
        """  
        training_data = list(training_data)  
        #将训练数据集强转为list  
        n = len(training_data)  
        cost = 0
        for j in range(epochs):   
            cost = self.update_mini_batch(training_data, eta)  
                #调用梯度下降算法  
            accuracy = self.evaluate(training_data)
            print("Epoch {} complete the loss is {},and  the acc is{}".format(j,cost,accuracy))        
   
    def update_mini_batch(self, mini_batch, eta):         
        """        
         基于反向传播的简单梯度下降算法更新网络的权重和偏置 
        """  
        nabla_b = [np.zeros(b.shape) for b in self.biases]  
        #for b in self.biases:  
            #print("b.shape=", b.shape)    
        nabla_w = [np.zeros(w.shape) for w in self.weights]  
        loss = 0
        for x, y in mini_batch:  
            delta_nabla_b, delta_nabla_w ,loss = self.backprop(x, y)  
            loss +=loss
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  
        self.weights = [w-(eta/len(mini_batch))*nw  
                        for w, nw in zip(self.weights, nabla_w)]  
        self.biases = [b-(eta/len(mini_batch))*nb  
                       for b, nb in zip(self.biases, nabla_b)]  
        return loss
  
  
    def backprop(self, x, y):  
    # 反向传播的算法，一种快速计算代价函数的梯度的方法。  
        nabla_b = [np.zeros(b.shape) for b in self.biases]  
        nabla_w = [np.zeros(w.shape) for w in self.weights]  
        # feedforward  
        activation = x  
        activations = [x] # list to store all the activations, layer by layer  
        zs = [] # list to store all the z vectors, layer by layer  
        for b, w in zip(self.biases, self.weights):  
            z = np.dot(w, activation)+b  
            zs.append(z)  
            activation = sigmoid(z)  
            activations.append(activation)  

        loss = abs(self.cost_derivative(activations[-1],y))
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])  
        nabla_b[-1] = delta  
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  
        for l in range(2, self.num_layers):  
            z = zs[-l]
            sp = sigmoid_prime(z)  
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  #a=w^T*delta 
            nabla_b[-l] = delta  
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  #w = dalta*a^T
        
        return (nabla_b, nabla_w,loss)  
  
    def evaluate(self, test_data):   
        right_number=0
        for x,y in test_data:
            y_= self.feedforward2(x)
           #print(self.feedforward(x),y)
            if y_==y:
                right_number=right_number+1;
        return  right_number/len(test_data)
  
    def cost_derivative(self, output_activations, y):  
        return (output_activations-y)  

    def get_sign(self,data):
        result = [self.feedforward(x) for x in data]
        return result 

  
#### activation functions  
def sigmoid(z):
    return 1.716*np.tanh(2/3*z)
  
def sigmoid_prime(z):  
    #计算 σ函数的导数  
    """Derivative of the sigmoid function."""  
    return 1.716*(1-np.tanh(2/3*z)**2)*2/3 
