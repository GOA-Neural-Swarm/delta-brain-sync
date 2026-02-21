import numpy as np
import pandas as pd
from scipy.stats import norm

def predict_expression(gene_expression, promoter_strength):
    # Calculate the probability of gene expression based on promoter strength
    probability = norm.cdf(promoter_strength, loc=0, scale=1)
    return probability

# Load the neon DNA sequence data
neon_data = pd.read_csv('neon_dna_sequence.csv')

# Extract the promoter strength values
promoter_strengths = neon_data['promoter_strength'].values

# Calculate the probability of gene expression for each promoter strength value
expression_probabilities = [predict_expression(gene_expression, promoter_strength) for promoter_strength in promoter_strengths]

# Save the results to a new CSV file
pd.DataFrame(expression_probabilities).to_csv('expression_probabilities.csv', index=False)