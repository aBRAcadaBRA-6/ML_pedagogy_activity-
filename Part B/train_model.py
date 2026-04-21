import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

print("Loading data...")

df = pd.read_csv(
    "./data_set/sfpark_filtered_136_247_100taxis.csv",
    sep=';',
    parse_dates=['timestamp']
)

# cleaning
df = df[df['capacity'] > 0]

# sort for time consistency
df = df.sort_values(['segmentid', 'timestamp'])

# features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

df['availability'] = df['capacity'] - df['occupied']

# =======================
# LAG FEATURES (KEY)
# =======================
df['lag_1'] = df.groupby('segmentid')['availability'].shift(1)
df['lag_2'] = df.groupby('segmentid')['availability'].shift(2)



df['rolling_mean_3'] = (
    df.groupby('segmentid')['availability']
    .shift(1)
    .rolling(3)
    .mean()
)

# drop missing from lag
df = df.dropna()
df = df.reset_index(drop=True)

# =======================
# DATASET
# =======================
features = [
    'segmentid', 'capacity', 'hour', 'dayofweek',
    'lag_1', 'lag_2', 'rolling_mean_3'
]
target = 'availability'

X = df[features]
y = df[target]

# split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("Predicting...")

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print(f"\nMAE with lag features: {mae:.4f}")


importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:\n")
print(importance)


joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")