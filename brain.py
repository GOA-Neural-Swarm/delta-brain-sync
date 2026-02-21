import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

# Load DNA sequence data
dna_seq = pd.read_csv('dna_sequence.csv')

# Preprocess data
dna_seq['sequence'] = dna_seq['sequence'].apply(lambda x: np.fromstring(x, dtype=int))
dna_seq['sequence'] = dna_seq['sequence'].apply(lambda x: x.tolist())

# Vectorize DNA sequence data
vectorizer = DictVectorizer()
dna_seq_vectorized = vectorizer.fit_transform(dna_seq['sequence'])

# Scale data
scaler = StandardScaler()
dna_seq_scaled = scaler.fit_transform(dna_seq_vectorized.toarray())

# Perform machine learning analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dna_seq_scaled, dna_seq['target'], test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))