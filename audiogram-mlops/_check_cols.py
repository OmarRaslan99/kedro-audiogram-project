import pickle

with open("data/05_model_input/X_train_vocal.pkl", "rb") as f:
    X = pickle.load(f)

print("COLUMNS:", list(X.columns))
print("SHAPE:", X.shape)
