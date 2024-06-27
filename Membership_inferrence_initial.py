import numpy as np

# Simple Neural Network Implementation
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)
    
    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        return self.z2

    def backward(self, X, y, output, learning_rate=0.01):
        output_error = output - y
        dW2 = self.a1.T.dot(output_error)
        db2 = np.sum(output_error, axis=0)
        
        hidden_error = output_error.dot(self.W2.T) * (1 - np.tanh(self.z1)**2)
        dW1 = X.T.dot(hidden_error)
        db1 = np.sum(hidden_error, axis=0)
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

np.random.seed(42)
X_train = np.random.randn(100, 3)
y_train = np.random.randn(100, 1)

# Train the model
model = SimpleNN(input_size=3, hidden_size=5, output_size=1)
model.train(X_train, y_train, epochs=5000, learning_rate=0.01)

# Model Inversion
def model_inversion_attack(model, target_output, learning_rate=0.1, epochs=500):
    inverted_input = np.random.randn(1, 3)  # Random initial input
    for _ in range(epochs):
        output = model.forward(inverted_input)
        loss = np.mean((output - target_output) ** 2)
        output_error = output - target_output
        
        hidden_error = output_error.dot(model.W2.T) * (1 - np.tanh(model.z1)**2)
        
        inverted_input_grad = hidden_error.dot(model.W1.T)
        
        inverted_input -= learning_rate * inverted_input_grad
    return inverted_input

target_output = model.forward(X_train[0:1])
inverted_input = model_inversion_attack(model, target_output)
print("Original Input:", X_train[0:1])
print("Inverted Input:", inverted_input)

# Membership Inference Attack
class MembershipInferenceAttack:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.forward(X)

def membership_inference_attack(model, shadow_model, data_point):
    shadow_predictions = shadow_model.predict(data_point)
    model_predictions = model.forward(data_point)
    
    return np.mean(np.abs(model_predictions - shadow_predictions))

shadow_model = SimpleNN(input_size=3, hidden_size=5, output_size=1)
shadow_model.train(X_train, y_train, epochs=5000, learning_rate=0.01)

data_point = X_train[0:1]
membership_score = membership_inference_attack(model, shadow_model, data_point)
print("Membership Score:", membership_score)
