import pandas as pd

print("Program started")

df = pd.read_csv(
    "./data_set/sfpark_filtered_136_247_100taxis.csv",
    sep=';'
)

# cleaning
df = df[df['capacity'] > 0]

# convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Time analysis:\n")

print("Start time:", df['timestamp'].min())
print("End time:", df['timestamp'].max())

# extract time features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

print("\nUnique hours:", sorted(df['hour'].unique()))
print("Unique days of week:", sorted(df['dayofweek'].unique()))