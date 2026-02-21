import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load dataset
df = pd.read_csv('neon_dna_sequence_analysis.csv')

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['target'], axis=1))
y = df['target']

# Model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

# Train model
history = model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.show()

# Make predictions
y_pred = model.predict(X_scaled)

# Calculate AUC-ROC
auc = stats.roc_auc_score(y, y_pred)
print(f'AUC-ROC: {auc:.4f}')

# Plot ROC curve
fpr, tpr, _ = stats.roc_curve(y, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()