import pandas as pd

print("Loading data...")

df = pd.read_csv(
    "./data_set/sfpark_filtered_136_247_100taxis.csv",
    sep=';',
    parse_dates=['timestamp']
)

# =======================
# CLEANING
# =======================
df = df[df['capacity'] > 0]

# =======================
# FEATURE ENGINEERING
# =======================
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# target
df['availability'] = df['capacity'] - df['occupied']

# =======================
# FINAL FEATURES
# =======================
features = ['segmentid', 'capacity', 'hour', 'dayofweek']
target = 'availability'

X = df[features]
y = df[target]

print("\nDataset ready:\n")
print(X.head())
print("\nTarget:\n")
print(y.head())

print("\nShape:", X.shape)