import numpy as np
import tensorflow as tf

class RNA_QT45_Predator_Logic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.l2 = tf.keras.layers.Dense(64)
        self.l3 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x)

RNA_QT45_Predator = RNA_QT45_Predator_Logic()
RNA_QT45_Predator.compile(optimizer='adam', loss='mean_squared_error')

# Training data
X_train = np.random.rand(100, 10)
Y_train = np.random.rand(100, 1)

# Training
RNA_QT45_Predator.fit(X_train, Y_train, epochs=100)

# Predict
X_test = np.random.rand(10, 10)
Y_test = RNA_QT45_Predator.predict(X_test)

print(Y_test)