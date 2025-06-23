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
        return cost[0,0] #to have a scalar value
    
    def _gradient(self,X_b,y,theta):
        """Computing the Gradient of the Cost Function.
        Args:
            X_b (np.ndarray): Input features with bias term.
            y (np.ndarray): Target labels.
            theta (np.ndarray): Model parameters (weights). Shape (n+1, 1).
        Returns:
            np.ndarray: Gradient of the cost function.
        """
        m=len(y)
        if m == 0:
            return np.zeros_like(theta)
        h =self.sigmoid(X_b @ theta)
        gradient = 1/m*(X_b.T@(h-y))
        return gradient
    
    def Batch_gradient_Descent(self,X_b,y,theta_initial,learning_rate=0.01,epochs=1000):
        """Batch Gradient Descent for Logistic Regression.
        Args:
            X_b (np.ndarray): Input features with bias term.
            y (np.ndarray): Target labels.
            theta (np.ndarray): Initial model parameters (weights).
            learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            epochs (int, optional): Number of iterations for training. Defaults to 1000.
        Returns:
            np.ndarray: Final model parameters (weights).
        """
        theta=np.copy(theta_initial)
        self.cost_history = []  # Reset cost history for each training session
        
        for epoch in range(epochs):
            gradient = self._gradient(X_b, y , theta)
            theta = theta - learning_rate * gradient
            cost = self.cost_function(X_b, y, theta)
            self.cost_history.append(cost)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
        
        self.theta = theta
        return theta
    
    def stochastic_gradient_descent(self, X_b, y, theta_initial, learning_rate, n_epochs, m_samples):
        """
        Performs Stochastic Gradient Descent (SGD).
        Updates theta using the gradient computed on a single, randomly chosen training example at each step.

        Args:
            X_b (np.ndarray): Design matrix.
            y (np.ndarray): Target values.
            theta_initial (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate.
            n_epochs (int): Number of passes over the entire dataset.
            m_samples (int): Total number of training samples.

        Returns:
            np.ndarray: Optimized parameters (theta).
        """
        theta = np.copy(theta_initial)
        self.cost_history = [] 

        for epoch in range(n_epochs):
            indices = np.random.permutation(m_samples)
            X_b_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for i in range(m_samples):
                xi = X_b_shuffled[i:i+1] #As 2D array
                yi = y_shuffled[i:i+1]   #As 2D array
                gradient = self._gradient(xi, yi, theta) # Gradient for one sample
                theta = theta - learning_rate * gradient

            
            cost_epoch = self.cost_function(X_b, y, theta)
            self.cost_history.append(cost_epoch)
        return theta
    
    
    
    def mini_batch_gradient_descent(self, X_b, y, theta_initial, learning_rate, n_iterations, batch_size, m_samples):
        """
        Performs Mini-Batch Gradient Descent.
        Updates theta using the gradient computed on a small batch of training examples.

        Args:
            X_b (np.ndarray): Design matrix.
            y (np.ndarray): Target values.
            theta_initial (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate.
            n_iterations (int): Number of batch updates (iterations).
            batch_size (int): Size of each mini-batch.
            m_samples (int): Total number of training samples.

        Returns:
            np.ndarray: Optimized parameters (theta).
        """
        theta = np.copy(theta_initial)
        self.cost_history = [] # Reset cost history
        # n_batches_per_epoch = int(np.ceil(m_samples / batch_size)) # If using epochs

        for iteration in range(n_iterations):
            random_indices = np.random.choice(m_samples, batch_size, replace=False)
            X_batch = X_b[random_indices]
            y_batch = y[random_indices]

            gradient = self._gradient(X_batch, y_batch, theta) # Gradient for the mini-batch
            theta = theta - learning_rate * gradient

            # Calculate and store cost at each iteration (on full dataset for consistent tracking)
            cost_iteration = self.cost_function(X_b, y, theta)
            self.cost_history.append(cost_iteration)
        return theta
    
    def predict(self, X_b):
        """Predicting the class labels using the learned model parameters.
        Args:
            X_b (np.ndarray): Input features with bias term.
        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        if self.theta is None:
            raise ValueError("Model has not been trained yet.")
        
        probabilities = self.sigmoid(X_b @ self.theta)
        return (probabilities >= 0.5).astype(int) #return 0 or 1 based on threshold of 0.5
    
    def fit(self, X_b, y, solver="batch_gd", learning_rate=0.01, n_iterations=1000,
            n_epochs=50, batch_size=32, theta_initial=None):
        """
        Fits the Logistic regression model using the specified solver.

        Args:
            X_b (np.ndarray): Design matrix (features + intercept column, x0=1).
            y (np.ndarray): Target values.
            solver (str): The optimization algorithm.
                          Options:  "batch_gd", "sgd", "mini_batch_gd".
            learning_rate (float): Learning rate for Gradient Descent variants.
            n_iterations (int): Number of iterations (for BGD, Mini-Batch GD if not using epochs).
            n_epochs (int): Number of epochs (for SGD, or can be used for Mini-Batch GD).
            batch_size (int): Batch size for Mini-Batch GD.
            theta_initial (np.ndarray, optional): Initial guess for parameters [bias, weight1,...].
                                                  If None, initialized to zeros.

        Raises:
            ValueError: If an unsupported solver is specified.
        """
        if theta_initial is None:
            # Initialize theta with zeros. X_b.shape[1] is num_features + 1 (for intercept).
            self.theta = np.zeros((X_b.shape[1], 1))
        else:
            self.theta = np.copy(theta_initial)

        m_samples = len(y) # Total number of training examples

        if solver == "batch_gd":
            self.theta = self.Batch_gradient_Descent(X_b, y, self.theta, learning_rate, n_iterations)
        elif solver == "sgd":
            self.theta = self.stochastic_gradient_descent(X_b, y, self.theta, learning_rate, n_epochs, m_samples)
        elif solver == "mini_batch_gd":
            self.theta = self.mini_batch_gradient_descent(X_b, y, self.theta, learning_rate, n_iterations, batch_size, m_samples)
        else:
            raise ValueError(f"Unsupported solver: {solver}. Choose from 'batch_gd', 'sgd', 'mini_batch_gd'.")
    
        