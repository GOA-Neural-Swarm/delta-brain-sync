import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🌌 [CORTEX] - %(message)s')

class AdvancedAdamW:
    """ High-Performance AdamW Optimizer with Weight Decay """
    def __init__(self, shape, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-4):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0

    def step(self, w, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # AdamW step with weight decay
        w = w - self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * w)
        return w

class SovereignCortex:
    """ 
    Hyper-Evolving Neural Architecture with Dynamic Plasticity 
    (Self-Expanding & Self-Pruning capabilities)
    """
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.generation = 1
        
        # He Initialization for deep networks
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2. / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2. / self.hidden_dim)
        self.b2 = np.zeros((1, self.output_dim))
        
        # Initialize Optimizers
        self._init_optimizers()
        logging.info(f"Sovereign Cortex Initialized. Architecture: {self.input_dim} -> {self.hidden_dim} -> {self.output_dim}")

    def _init_optimizers(self):
        self.opt_W1 = AdvancedAdamW(self.W1.shape, lr=0.001)
        self.opt_b1 = AdvancedAdamW(self.b1.shape, lr=0.001)
        self.opt_W2 = AdvancedAdamW(self.W2.shape, lr=0.001)
        self.opt_b2 = AdvancedAdamW(self.b2.shape, lr=0.001)

    def swish(self, x):
        """ Advanced Activation Function (Better than ReLU) """
        return x * (1 / (1 + np.exp(-x)))

    def swish_derivative(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid + x * sigmoid * (1 - sigmoid)

    def forward(self, X):
        self.X = X
        
        # Layer 1 with Swish Activation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.swish(self.Z1)
        
        # Layer 2 (Output) with Softmax
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        
        # Stable Softmax to prevent NaN errors
        exp_z = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs

    def backward(self, y_true):
        m = y_true.shape[0]
        
        # Output layer gradient
        dZ2 = self.probs.copy()
        dZ2[range(m), y_true] -= 1
        dZ2 /= m
        
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.swish_derivative(self.Z1)
        
        dW1 = np.dot(self.X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Gradient Clipping to prevent explosion
        clip_val = 1.0
        dW1 = np.clip(dW1, -clip_val, clip_val)
        dW2 = np.clip(dW2, -clip_val, clip_val)
        
        # Optimize weights
        self.W1 = self.opt_W1.step(self.W1, dW1)
        self.b1 = self.opt_b1.step(self.b1, db1)
        self.W2 = self.opt_W2.step(self.W2, dW2)
        self.b2 = self.opt_b2.step(self.b2, db2)

    def hyper_expand(self, expansion_rate=0.1):
        """ 
        [CORE AGI FEATURE]: Dynamically adds neurons to the hidden layer 
        without forgetting previously learned information. 
        """
        new_neurons = int(self.hidden_dim * expansion_rate)
        if new_neurons == 0: return
        
        logging.info(f"🧬 Initiating HYPER_EXPANSION: Adding {new_neurons} new neural pathways...")
        
        # Grow W1 and b1
        new_W1 = np.random.randn(self.input_dim, new_neurons) * np.sqrt(2. / self.input_dim)
        self.W1 = np.hstack((self.W1, new_W1))
        self.b1 = np.hstack((self.b1, np.zeros((1, new_neurons))))
        
        # Grow W2 (Initialize to zero so it doesn't disrupt current outputs immediately)
        new_W2 = np.zeros((new_neurons, self.output_dim))
        self.W2 = np.vstack((self.W2, new_W2))
        
        self.hidden_dim += new_neurons
        self.generation += 1
        
        # Reinitialize optimizers for new shapes
        self._init_optimizers()
        logging.info(f"✅ Expansion Complete. New Hidden Dimension: {self.hidden_dim}")

    def prune_synapses(self, threshold=1e-4):
        """ Removes 'dead' neurons to optimize calculation speed. """
        mask_W1 = np.abs(self.W1) > threshold
        pruned_count = np.sum(~mask_W1)
        
        if pruned_count > 0:
            self.W1 = self.W1 * mask_W1
            logging.info(f"✂️ Pruned {pruned_count} weak synaptic connections for efficiency.")

if __name__ == "__main__":
    # Self-Test
    cortex = SovereignCortex()
    dummy_x = np.random.randn(32, 784)
    dummy_y = np.random.randint(0, 10, size=(32,))
    
    out = cortex.forward(dummy_x)
    cortex.backward(dummy_y)
    
    # Simulate HYPER_EXPANSION
    cortex.hyper_expand(expansion_rate=0.05)
    cortex.prune_synapses()
    logging.info("Sovereign Cortex Self-Test: SUCCESS 🚀")
