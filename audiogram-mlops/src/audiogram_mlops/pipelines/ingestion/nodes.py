import pandas as pd

def clean_tonal_exams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertit tout en numérique (ex: "E" -> NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprime lignes avec valeurs manquantes
    df = df.dropna()

    # Garde-fou simple : dB dans une plage plausible
    df = df[(df >= 0).all(axis=1) & (df <= 120).all(axis=1)]

    df = df.reset_index(drop=True)
    return df
