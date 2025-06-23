import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """Logistic Regression Implementation from Scratch.
    """
    def __init__(self):
        """
        Intialize the Logistic Regression model.
        theta is a column vector containing the model parameters.
        Cost history is used to track the cost during training during each iteration/epoch.
        """
        self.theta = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Computing the Sigmoid Function.
        Args:
            z (np.ndarray): Input to the sigmoid function.
        Returns:
            np.ndarray: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
        
    def cost_function(self,X_b,y,theta):
        """Computing the Cost Function for Logistic Regression.(Binary Cross-Entropy Loss)
        Args:
            X_b (np.ndarray): Input features with bias term.
            y (np.ndarray): Target labels.
            theta (np.ndarray): Model parameters (weights). Shape (n+1, 1).
        Returns:
            float: The computed cost.
        """
        m= len(y)
        if m == 0:
            return 0.0
        h = self.sigmoid(X_b @ theta)
        cost= -1/m * (y.T @ np.log(h + 1e-7) + (1 - y).T @ np.log(1 - h + 1e-7)) #to prevent log(0)
        self.cost_history.append(cost)
        return cost[0,0] #to have a scalar value
    
    