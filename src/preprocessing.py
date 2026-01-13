import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path: str):
    df = pd.read_csv(path)

    X = df.drop(columns=["customer_id", "default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
