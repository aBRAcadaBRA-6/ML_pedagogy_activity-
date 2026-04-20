import pandas as pd
import matplotlib.pyplot as plt

print("Loading data...")

df = pd.read_csv(
    "./data_set/sfpark_filtered_136_247_100taxis.csv",
    sep=';',
    parse_dates=['timestamp']
)

# basic cleaning
df = df[df['capacity'] > 0]

print("Creating features...")

df['hour'] = df['timestamp'].dt.hour

print("Plotting...")

hourly_avg = df.groupby('hour')['occupied'].mean()

hourly_avg.plot()

plt.title("Average Occupancy by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Occupied Spots")

plt.savefig("average_occupancy_by_hour.png")    