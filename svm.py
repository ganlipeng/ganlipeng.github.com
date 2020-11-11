import numpy as np

class SVM:
    def __init__(self,learning_rate=0.01,lambda_para=0.001,n_iter=10000):
        self.lr = learning_rate
        self.lambda_para = lambda_para
        self.n_iter = n_iter
        self.w = None
        self.b = None
        
        
        
        
    def fit(self,X,y):
        
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iter):
            
            for i,x_i in enumerate(X):
                condition = y[i]*(np.dot(x_i, self.w) + self.b)>=1
                if condition:
                    self.w -= self.lr*(2*self.lambda_para*self.w)
                    
                else:
                    self.w -= self.lr*((2*self.lambda_para*self.w) - np.dot(x_i,y[i]))
                    self.b -= - self.lr*(y[i])
                    
            
                