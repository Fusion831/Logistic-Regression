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