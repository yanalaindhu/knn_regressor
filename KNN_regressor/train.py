import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("data/insurance.csv")

# ---------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------
le = LabelEncoder()

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------
X = df.drop("charges", axis=1)

y = df["charges"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
model = KNeighborsRegressor(
    n_neighbors=5
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------
pickle.dump(
    model,
    open("models/knn_model.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

print("Model Saved Successfully")