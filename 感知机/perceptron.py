class perceptron:
    '''
    perceptron class, use Stochastic Gradient Descent to training coefficient
    '''
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.w = np.ones((len(x_train[0])))
        self.b = np.ones((1))
    def sign(self, x_input):
        y = np.matmul(x_input, self.w.T) + self.b
        y = np.array([-1 if v < 0 else 1 for v in y])
        return y
    def sgd(self,learn_rate = 0.01):
        '''
        Stochastic Gradient Descent training function
        default learning rate learn_rate = 0.01
        '''
        stop_flag = False
        while not stop_flag:
            wrong_count = 0
            y = self.sign(self.x_train)
            for i in range(len(self.y_train)):
                if y[i]*y_train[i] < 0:
                    self.w += learn_rate*x_train[i]*y_train[i]
                    self.b += learn_rate*y_train[i]
                    wrong_count += 1
            if wrong_count == 0:
                stop_flag = True
    def loss_func(self):
        loss = 0
        y = self.sign(self.x_train)*self.y_train
        for l in y:
            if l < 0:
                loss += l
        return loss
