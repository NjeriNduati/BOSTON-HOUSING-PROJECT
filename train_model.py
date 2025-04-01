import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Preprocessing
X_full = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # All features
y = raw_df.values[1::2, 2]  # Target variable

# Selecting significant features: RM (5th column), LSTAT (13th column), CRIM (1st column)
X = X_full[:, [5, 12, 0]]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
with open('linear_model.pkl', 'wb') as file:
    pickle.dump(model, file)

