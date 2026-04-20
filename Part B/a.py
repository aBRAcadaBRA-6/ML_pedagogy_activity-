import pandas as pd

print("Program started")

df = pd.read_csv(
    "./data_set/sfpark_filtered_136_247_100taxis.csv",
    sep=';'
)

# Apply previous cleaning
df = df[df['capacity'] > 0]

print("Data cleaned")

print("\nValidation checks:\n")

# 1. occupied > capacity (invalid)
invalid_occupied = (df['occupied'] > df['capacity']).sum()
print(f"occupied > capacity: {invalid_occupied}")

# 2. negative occupied
negative_occupied = (df['occupied'] < 0).sum()
print(f"negative occupied: {negative_occupied}")

# 3. basic stats
print("\nBasic stats:\n")
print(df[['capacity', 'occupied']].describe())