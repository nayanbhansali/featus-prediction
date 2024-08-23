import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
dataset_url = "fetal_health.csv"
df = pd.read_csv(dataset_url)

# Define feature columns
features = ['abnormal_short_term_variability', 'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability', 'histogram_mean', 'histogram_mode']

# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
train_X = train[features]
train_y = train['fetal_health']

# Train model
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(train_X, train_y)

# Save model to a file
joblib.dump(model, 'model.joblib')

print("Model saved to model.joblib")
