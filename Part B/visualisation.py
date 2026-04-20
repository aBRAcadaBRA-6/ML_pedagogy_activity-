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

# IMPORTANT: normalized metric
df['occupancy_rate'] = df['occupied'] / df['capacity']

# =======================
# 1. HOURLY PATTERN
# =======================
hourly_avg = df.groupby('hour')['occupancy_rate'].mean()

hourly_avg.plot()
plt.title("Average Occupancy Rate by Hour")
plt.xlabel("Hour")
plt.ylabel("Occupancy Rate")
plt.savefig("average_occupancy_rate_by_hour.png")
plt.clf()

peak_hour = hourly_avg.idxmax()
low_hour = hourly_avg.idxmin()

# =======================
# 2. DAY-OF-WEEK PATTERN
# =======================
daily_avg = df.groupby('dayofweek')['occupancy_rate'].mean()

daily_avg.plot(kind='bar')
plt.title("Average Occupancy Rate by Day of Week")
plt.xlabel("Day (0=Mon)")
plt.ylabel("Occupancy Rate")
plt.savefig("average_occupancy_rate_by_day.png")
plt.clf()

max_day = daily_avg.idxmax()
min_day = daily_avg.idxmin()

# =======================
# 3. HOUR × DAY INTERACTION
# =======================
pivot = df.pivot_table(
    values='occupancy_rate',
    index='hour',
    columns='dayofweek',
    aggfunc='mean'
)

plt.imshow(pivot, aspect='auto')
plt.colorbar(label='Occupancy Rate')
plt.title("Occupancy Rate Heatmap (Hour vs Day)")
plt.xlabel("Day (0=Mon)")
plt.ylabel("Hour")
plt.savefig("heatmap_hour_day_rate.png")
plt.clf()

max_location = pivot.stack().idxmax()
min_location = pivot.stack().idxmin()

# =======================
# FINAL INSIGHTS (DETAILED)
# =======================
peak_h, peak_d = max_location
low_h, low_d = min_location

print("\n================ EDA INSIGHTS ================\n")

# ---- HOURLY ----
print("1. HOURLY PATTERN:")
print(f"- Peak occupancy occurs at {peak_hour}:00 hours.")
print(f"- Lowest occupancy occurs at {low_hour}:00 hours.")
print("- Interpretation: Parking demand increases through the day, peaks in the evening,")
print("  and drops significantly during early morning hours.\n")

# ---- DAILY ----
print("2. WEEKLY PATTERN:")
print(f"- Highest demand occurs on {calendar.day_name[max_day]}.")
print(f"- Lowest demand occurs on {calendar.day_name[min_day]}.")
print("- Interpretation: Weekdays (especially Friday) have higher parking demand,")
print("  while weekends (especially Sunday) show reduced activity.\n")

# ---- INTERACTION ----
print("3. TIME + DAY INTERACTION:")
print(f"- Peak demand occurs at {peak_h}:00 on {calendar.day_name[peak_d]}.")
print(f"- Lowest demand occurs at {low_h}:00 on {calendar.day_name[low_d]}.")
print("- Interpretation: The strongest parking pressure is during Friday evenings,")
print("  while early mornings on weekends show the lowest utilization.\n")

# ---- SUMMARY ----
print("4. OVERALL CONCLUSION:")
print("- Parking availability is strongly influenced by time-of-day and day-of-week.")
print("- Evening hours and weekdays drive high occupancy.")
print("- Early mornings and weekends show low demand.")
print("- These patterns justify using time-based features in ML models.\n")

print("=============================================\n")