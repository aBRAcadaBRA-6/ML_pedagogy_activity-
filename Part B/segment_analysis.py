import pandas as pd
import matplotlib.pyplot as plt

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
df['occupancy_rate'] = df['occupied'] / df['capacity']

# =======================
# SEGMENT ANALYSIS
# =======================

# average occupancy per segment
segment_avg = df.groupby('segmentid')['occupancy_rate'].mean()

# sort segments
segment_sorted = segment_avg.sort_values()

# top 10 busiest
top_10 = segment_sorted.tail(10)

# bottom 10 least used
bottom_10 = segment_sorted.head(10)

# =======================
# PLOTTING
# =======================

# Top 10
top_10.plot(kind='barh')
plt.title("Top 10 Most Occupied Segments")
plt.xlabel("Occupancy Rate")
plt.savefig("top_10_segments.png")
plt.clf()

# Bottom 10
bottom_10.plot(kind='barh')
plt.title("Top 10 Least Occupied Segments")
plt.xlabel("Occupancy Rate")
plt.savefig("bottom_10_segments.png")
plt.clf()

# =======================
# DISTRIBUTION
# =======================
segment_avg.plot(kind='hist', bins=50)
plt.title("Distribution of Segment Occupancy Rates")
plt.xlabel("Occupancy Rate")
plt.savefig("segment_distribution.png")
plt.clf()

# =======================
# INSIGHTS
# =======================
print("\n=========== SEGMENT INSIGHTS ===========\n")

print("Top 5 busiest segments:")
print(top_10.sort_values(ascending=False).head(5))

print("\nTop 5 least used segments:")
print(bottom_10.sort_values().head(5))

print("\nGeneral Observations:")
print("- Some segments are consistently near full occupancy.")
print("- Some segments are rarely used.")
print("- There is high variability across locations.")
print("- Segment ID is a strong predictive feature.\n")

print("=======================================\n")