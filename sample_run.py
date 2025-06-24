from sklearn.datasets import load_breast_cancer
from Logistic_Regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
X, y = load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler() 

model = LogisticRegression()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)       
model.fit(X_train, y_train, solver="batch_gd", learning_rate=0.01, n_iterations=1000)


predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")



plt.plot(model.cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()