import pandas as pd
import numpy as np

# Generate random data
np.random.seed(0)
n_samples = 1000  # Change this to 1000
data = {
    'loan_id': np.arange(1, n_samples + 1),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'married': np.random.choice(['Yes', 'No'], n_samples),
    'dependents': np.random.choice([0, 1, 2, '3+'], n_samples),
    'education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
    'self_employed': np.random.choice(['Yes', 'No'], n_samples),
    'applicant_income': np.random.randint(1500, 10000, n_samples),
    'coapplicant_income': np.random.randint(0, 5000, n_samples),
    'loan_amount': np.random.randint(50, 500, n_samples),
    'loan_amount_term': np.random.choice([120, 180, 240, 360], n_samples),
    'credit_history': 1,  # Set credit history to 1 (indicating good credit history) for all samples
    'property_area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples),
    'loan_status': np.random.choice(['Y', 'N'], n_samples, p=[0.5, 0.5])  # Set loan status to 'Y' or 'N' with equal probability
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('Loan_Applications_Sample.csv', index=False)

# Print the DataFrame
print(df.head())
