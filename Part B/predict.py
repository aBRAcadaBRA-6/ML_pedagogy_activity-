import pandas as pd
import joblib

print("Loading model and data...")

# Load model
model = joblib.load("model.pkl")

# Load dataset
df = pd.read_csv(
    "./data_set/sfpark_filtered_136_247_100taxis.csv",
    sep=';',
    parse_dates=['timestamp']
)

# =======================
# PREP
# =======================
df = df[df['capacity'] > 0]
df = df.sort_values(['segmentid', 'timestamp'])

df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['availability'] = df['capacity'] - df['occupied']

valid_segments = df['segmentid'].unique()

# =======================
# DISPLAY VALID RANGES
# =======================
print("\n=== VALID INPUT RANGES ===")
print(f"Segment IDs: {len(valid_segments)} available")
print(f"Example segment IDs: {list(valid_segments[:5])}")
print("Hour: 0 → 23")
print("Day of week: 0 (Mon) → 6 (Sun)")
print("==========================\n")

print("=== Parking Availability Prediction ===\n")

# =======================
# MODE SELECTION
# =======================
print("Select mode:")
print("1. Simple (recommended)")
print("2. Advanced")

mode = input("Enter choice (1/2): ").strip()

# =======================
# COMMON INPUT
# =======================
segmentid = int(input("Segment ID: "))
if segmentid not in valid_segments:
    print("Invalid segment ID")
    exit()

hour = int(input("Hour (0-23): "))
if not (0 <= hour <= 23):
    print("Invalid hour")
    exit()

dayofweek = int(input("Day (0=Mon, 6=Sun): "))
if not (0 <= dayofweek <= 6):
    print("Invalid day")
    exit()

segment_data = df[df['segmentid'] == segmentid]

# =======================
# SIMPLE MODE
# =======================
if mode == "1":
    print("\nUsing smart historical approximation...")

    context_data = segment_data[
        (segment_data['hour'] == hour) &
        (segment_data['dayofweek'] == dayofweek)
    ]

    if len(context_data) < 3:
        print("Not enough matching data, fallback to recent data...")
        context_data = segment_data.tail(3)
    else:
        context_data = context_data.sort_values('timestamp').tail(3)

    lag_1 = context_data.iloc[-1]['availability']
    lag_2 = context_data.iloc[-2]['availability']
    rolling_mean_3 = context_data['availability'].mean()
    capacity = context_data.iloc[-1]['capacity']

# =======================
# ADVANCED MODE
# =======================
elif mode == "2":
    print("\n=== ADVANCED INPUT ===")
    print("Provide recent availability values:\n")

    capacity = int(input("Capacity: "))
    lag_1 = float(input("Availability t-1: "))
    lag_2 = float(input("Availability t-2: "))
    lag_3 = float(input("Availability t-3: "))

    rolling_mean_3 = (lag_1 + lag_2 + lag_3) / 3

else:
    print("Invalid mode")
    exit()

# =======================
# PREDICTION
# =======================
input_data = pd.DataFrame([{
    'segmentid': segmentid,
    'capacity': capacity,
    'hour': hour,
    'dayofweek': dayofweek,
    'lag_1': lag_1,
    'lag_2': lag_2,
    'rolling_mean_3': rolling_mean_3
}])

prediction = model.predict(input_data)[0]

# =======================
# OUTPUT
# =======================
print("\n--- RESULT ---")
print(f"Predicted available spots: {prediction:.2f}")

''' print("\n--- DEBUG INFO ---")
print(f"lag_1: {lag_1}, lag_2: {lag_2}, rolling_mean_3: {rolling_mean_3:.2f}")'''