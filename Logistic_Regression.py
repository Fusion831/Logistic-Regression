import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """Logistic Regression Implementation from Scratch.
    
    """
    def __init__(self, lambda_val=0.0):
        """
        
            
        Intialize the Logistic Regression model.
        theta is a column vector containing the model parameters.
        Cost history is used to track the cost during training during each iteration/epoch.
        
        args:
            lambda_val (float): Regularization parameter. Default is 0.0 (no regularization).
            
        """
        self.theta = None
        self.cost_history = []
        self.lambda_val = lambda_val
        
    
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
        if self.lambda_val > 0:
            
            reg_term = (self.lambda_val / (2 * m)) * np.sum(np.square(theta[1:]))
            
            cost = cost + reg_term
        
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
        if self.lambda_val > 0:
            reg_term = (self.lambda_val / m) * theta
            reg_term[0] = 0
            gradient = gradient + reg_term
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
    
    def fit(self, X, y, solver="batch_gd", learning_rate=0.01, n_iterations=1000,
            n_epochs=50, batch_size=32, theta_initial=None):
        """
        Fits the Logistic regression model to the training data.

        This method preprocesses the input features by adding a bias term,
        initializes model parameters if not provided, and then calls the
        specified gradient descent solver to optimize the parameters.

        Args:
            X (np.ndarray): Input features (training data).
                            Shape (m_samples, n_features).
            y (np.ndarray): Target labels (training data).
                            Shape (m_samples,) or (m_samples, 1).
            solver (str, optional): The optimization algorithm to use.
                                    Options: "batch_gd", "sgd", "mini_batch_gd".
                                    Defaults to "batch_gd".
            learning_rate (float, optional): Learning rate for gradient descent.
                                             Defaults to 0.01.
            n_iterations (int, optional): Number of iterations for Batch GD and
                                          Mini-Batch GD (if not epoch-based).
                                          Defaults to 1000.
            n_epochs (int, optional): Number of full passes over the dataset (epochs).
                                      Used by SGD and can be used by Mini-Batch GD.
                                      Defaults to 50.
            batch_size (int, optional): Size of mini-batches for Mini-Batch GD.
                                        Defaults to 32.
            theta_initial (np.ndarray, optional): Initial guess for model parameters (theta),
                                                  including the bias term. Shape (n_features + 1, 1).
                                                  If None, parameters are initialized to zeros.
                                                  Defaults to None.

        Raises:
            ValueError: If an unsupported solver is specified or if theta_initial
                        has an incompatible shape.
        """
        self.solver_used_for_last_fit = solver
        self.cost_history = []  

        # Add bias term to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize theta if not provided, or use provided theta_initial
        if theta_initial is None:
            initial_theta_for_gd = np.zeros((X_b.shape[1], 1))
        else:
            if theta_initial.shape[0] != X_b.shape[1] or theta_initial.ndim != 2 or theta_initial.shape[1] != 1 :
                raise ValueError(
                    f"Shape of theta_initial {theta_initial.shape} must be ({X_b.shape[1]}, 1) "
                    f"to be compatible with X_b columns {X_b.shape[1]}"
                )
            initial_theta_for_gd = np.copy(theta_initial)

        # Ensure y is a column vector
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        m_samples = len(y) # Total number of training examples

        final_theta = None 

        
        if solver == "batch_gd":
            final_theta = self.Batch_gradient_Descent(X_b, y, initial_theta_for_gd, learning_rate, n_iterations)
        elif solver == "sgd":
            final_theta = self.stochastic_gradient_descent(X_b, y, initial_theta_for_gd, learning_rate, n_epochs, m_samples)
        elif solver == "mini_batch_gd":
            final_theta = self.mini_batch_gradient_descent(X_b, y, initial_theta_for_gd, learning_rate, n_iterations, batch_size, m_samples)
        else:
            raise ValueError(f"Unsupported solver: {solver}. Choose from 'batch_gd', 'sgd', 'mini_batch_gd'.")
        
        self.theta = final_theta 

    def predict(self, X, threshold=0.5):
        """
        Predicts class labels for input samples.

        This method first adds a bias term to the input features, then
        calculates the probabilities using the sigmoid function and the
        learned model parameters (theta). Finally, it applies a threshold
        to these probabilities to assign class labels (0 or 1).

        Args:
            X (np.ndarray): Input features for which to make predictions.
                            Shape (m_samples, n_features).
            threshold (float, optional): The threshold used to convert probabilities
                                         to class labels. If probability >= threshold,
                                         class 1 is predicted, otherwise class 0.
                                         Defaults to 0.5.

        Returns:
            np.ndarray: Predicted class labels (0 or 1) for each input sample.
                        Shape (m_samples, 1).

        Raises:
            ValueError: If the model has not been trained yet (i.e., theta is None).
        """
        if self.theta is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Add bias term to X for prediction
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate probabilities
        probabilities = self.sigmoid(X_b @ self.theta)
        
        # Apply threshold to get class labels
        return (probabilities >= threshold).astype(int)
    
    
    def plot_cost_history(self):
        """Plotting the Cost History during Training.
        This function plots the cost history to visualize the convergence of the model.
        """
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iterations/Epochs')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    
    
    
    
    def plot_decision_boundary(self, X, y, title="Decision Boundary"):
        """
        Plots the decision boundary for a 2D dataset.
        Assumes the model has been trained and X has 2 features.

        Args:
            X (np.ndarray): Feature matrix (m samples, 2 features).
                            This should be the same data (or scaled the same way)
                            that the model was trained on, but WITHOUT the bias column.
            y (np.ndarray): True labels (m samples,). Used for coloring data points.
            title (str): Title for the plot.
        """
        if self.theta is None:
            print("Model has not been trained yet. Call fit() first.")
            return
        if X.shape[1] != 2:
            print("Decision boundary plotting is only supported for 2D feature data.")
            return

        
        if y.ndim > 1:
            y = y.ravel()

        
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50)


        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x1_plot = np.linspace(x1_min, x1_max, 100) 

       
        theta0 = self.theta[0, 0]
        theta1 = self.theta[1, 0]
        theta2 = self.theta[2, 0]

        
        
        if theta2 != 0: 
            x2_plot = (-theta0 - theta1 * x1_plot) / theta2
            plt.plot(x1_plot, x2_plot, label='Decision Boundary', color='green', linewidth=2)
        elif theta1 != 0: 
            
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            plt.axvline(x=(-theta0 / theta1), color='green', linestyle='--', linewidth=2, label='Decision Boundary (Vertical)')
            plt.ylim(y_min, y_max)
        else: 
            print("Warning: Decision boundary might be ill-defined (theta1 and theta2 are zero or near zero).")
            


        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        