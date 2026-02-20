import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

class Brain:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.svm = SVC(kernel='linear', C=1, probability=True)

    def process_data(self):
        self.data = self.scaler.fit_transform(self.data)
        self.data = self.pca.fit_transform(self.data)
        X_train, X_test, y_train, y_test = train_test_split(self.data[:, :2], self.data[:, 2], test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_svm(self, X_train, y_train):
        self.svm.fit(X_train, y_train)

    def predict(self, X_test):
        return self.svm.predict(X_test)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Confusion Matrix:\n{cm}")

    def recursive_upgrade(self):
        self.data = self.process_data()
        self.train_svm(*self.data)
        y_pred = self.predict(*self.data[2:])
        self.evaluate(*self.data[2:], y_pred)

brain = Brain(np.array([[1, 2, 0], [3, 4, 1], [5, 6, 0]]))
brain.recursive_upgrade()