import pandas as pd
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()

X = digits.data
y = digits.target

# Create column names
columns = [f'pixel_{i}' for i in range(64)]

# Create DataFrame
df = pd.DataFrame(X, columns=columns)

# Add label column
df['label'] = y

# Display
print(df.head())