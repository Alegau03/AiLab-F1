# utils_gradio.py
import pandas as pd

def to_str_type(X_df: pd.DataFrame) -> pd.DataFrame:
    """Converts all columns in the input DataFrame to string type."""
    if not isinstance(X_df, pd.DataFrame):
        # Potremmo provare a convertirlo, ma è meglio assicurarsi che riceva un DataFrame
        raise TypeError(f"to_str_type si aspetta un DataFrame, ha ricevuto {type(X_df)}")
    return X_df.astype(str)


# Questa funzione è utile per convertire i dati in un formato compatibile con Gradio