import pandas as pd
import matplotlib.pyplot as plt
import calendar

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

# =======================
# 1. HOURLY PATTERN
# =======================
hourly_avg = df.groupby('hour')['occupied'].mean()

hourly_avg.plot()
plt.title("Average Occupancy by Hour")
plt.xlabel("Hour")
plt.ylabel("Avg Occupied")
plt.savefig("average_occupancy_by_hour.png")
plt.clf()

peak_hour = hourly_avg.idxmax()
low_hour = hourly_avg.idxmin()

# =======================
# 2. DAY-OF-WEEK PATTERN
# =======================
daily_avg = df.groupby('dayofweek')['occupied'].mean()

daily_avg.plot(kind='bar')
plt.title("Average Occupancy by Day of Week")
plt.xlabel("Day (0=Mon)")
plt.ylabel("Avg Occupied")
plt.savefig("average_occupancy_by_day.png")
plt.clf()

max_day = daily_avg.idxmax()
min_day = daily_avg.idxmin()

# =======================
# 3. HOUR × DAY INTERACTION
# =======================
pivot = df.pivot_table(
    values='occupied',
    index='hour',
    columns='dayofweek',
    aggfunc='mean'
)

plt.imshow(pivot, aspect='auto')
plt.colorbar(label='Avg Occupied')
plt.title("Occupancy Heatmap (Hour vs Day)")
plt.xlabel("Day (0=Mon)")
plt.ylabel("Hour")
plt.savefig("heatmap_hour_day.png")
plt.clf()

max_location = pivot.stack().idxmax()
min_location = pivot.stack().idxmin()

# =======================
# FINAL INSIGHTS
# =======================
print("\n--- INSIGHTS ---")

print(f"Peak hour: {peak_hour}")
print(f"Lowest hour: {low_hour}")

print(f"Highest occupancy day: {calendar.day_name[max_day]}")
print(f"Lowest occupancy day: {calendar.day_name[min_day]}")

peak_h, peak_d = max_location
low_h, low_d = min_location

print(f"Peak combination: Hour {peak_h}, {calendar.day_name[peak_d]}")
print(f"Lowest combination: Hour {low_h}, {calendar.day_name[low_d]}")