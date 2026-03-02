import pandas as pd
from sklearn.model_selection import train_test_split

BEFORE_COLS = [
    "before_exam_125_Hz","before_exam_250_Hz","before_exam_500_Hz",
    "before_exam_1000_Hz","before_exam_2000_Hz","before_exam_4000_Hz",
    "before_exam_8000_Hz",
]
AFTER_COLS = [
    "after_exam_125_Hz","after_exam_250_Hz","after_exam_500_Hz",
    "after_exam_1000_Hz","after_exam_2000_Hz","after_exam_4000_Hz",
    "after_exam_8000_Hz",
]

def make_train_test(df: pd.DataFrame):
    X = df[BEFORE_COLS].copy()
    y = df[AFTER_COLS].copy()

    return train_test_split(X, y, test_size=0.2, random_state=42)
