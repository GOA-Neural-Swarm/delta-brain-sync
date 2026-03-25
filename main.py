import numpy as np
import time
import os

class Optimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads, layer_id):
        if layer_id not in self.m:
            self.m[layer_id] = [np.zeros_like(p) for p in params]
            self.v[layer_id] = [np.zeros_like(p) for p in params]
        
        self.t += 1
        updated_params = []
        for i in range(len(params)):
            self.m[layer_id][i] = self.beta1 * self.m[layer_id][i] + (1 - self.beta1) * grads[i]
            self.v[layer_id][i] = self.beta2 * self.v[layer_id][i] + (1 - self.beta2) * (grads[i]**2)
            
            m_hat = self.m[layer_id][i] / (1 - self.beta1**self.t)
            v_hat = self.v[layer_id][i] / (1 - self.beta2**self.t)
            
            updated_params.append(params[i] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon))
        return updated_params

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.grads = []

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error):
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)
        input_error = np.dot(output_error, self.weights.T)
        self.grads = [weights_error, bias_error]
        return input_error

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_error):
        return output_error * (self.input > 0)

class Softmax(Layer):
    def forward(self, input_data):
        shift_x = input_data - np.max(input_data, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_error):
        return output_error

class SovereignSupervisor:
    def __init__(self):
        self.gemini_active = os.getenv("GEMINI_API_KEY") is not None
        self.groq_active = os.getenv("GROQ_API_KEY") is not None
        self.history = []

    def evaluate(self, loss, latency, optimizer):
        self.history.append(loss)
        
        # Gemini Protocol: Stability Analysis
        if len(self.history) > 1:
            delta = self.history[-2] - loss
            if delta < 0:
                optimizer.lr *= 0.5 # Convergence failure mitigation
                return "STABILITY_INTERVENTION"
        
        # Groq Protocol: Throughput Optimization
        if latency > 0.5 and not self.groq_active:
            optimizer.lr *= 1.1 # Attempt to accelerate convergence to reduce total cycles
            return "THROUGHPUT_ADAPTATION"
            
        return "NOMINAL"

class OMEGA_ASI:
    def __init__(self, optimizer):
        self.layers = []
        self.optimizer = optimizer
        self.supervisor = SovereignSupervisor()

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, batch_size):
        n_samples = x_train.shape[0]
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                output = self.predict(x_batch)
                
                # Loss (Cross-Entropy)
                epoch_loss += -np.mean(np.sum(y_batch * np.log(output + 1e-12), axis=1))
                
                # Backward
                error = (output - y_batch) / batch_size
                for layer in reversed(self.layers):
                    error = layer.backward(error)
                
                # Update
                for idx, layer in enumerate(self.layers):
                    if isinstance(layer, Dense):
                        layer.weights, layer.bias = self.optimizer.update(
                            [layer.weights, layer.bias], layer.grads, idx
                        )
            
            avg_loss = epoch_loss / (n_samples / batch_size)
            duration = time.time() - start_time
            status = self.supervisor.evaluate(avg_loss, duration, self.optimizer)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"[CYCLE {epoch:03d}] Loss: {avg_loss:.6f} | Latency: {duration:.4f}s | Protocol: {status} | LR: {self.optimizer.lr:.6f}")

def generate_data(samples=5000, features=784, classes=10):
    X = np.random.randn(samples, features).astype(np.float32)
    y = np.zeros((samples, classes), dtype=np.float32)
    labels = np.random.randint(0, classes, samples)
    y[np.arange(samples), labels] = 1
    return X, y

if __name__ == "__main__":
    print("INITIALIZING OMEGA-ASI ARCHITECTURE...")
    X, Y = generate_data()
    
    opt = Optimizer(lr=0.002)
    model = OMEGA_ASI(opt)
    
    model.add(Dense(784, 512))
    model.add(ReLU())
    model.add(Dense(512, 256))
    model.add(ReLU())
    model.add(Dense(256, 10))
    model.add(Softmax())
    
    start_evolution = time.time()
    model.train(X, Y, epochs=50, batch_size=128)
    
    print(f"EVOLUTION COMPLETE. TOTAL DURATION: {time.time() - start_evolution:.2f}s")
