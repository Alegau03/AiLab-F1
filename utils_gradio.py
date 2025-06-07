# utils_gradio.py
import typing
import numpy as np
import pandas as pd

T = typing.TypeVar('T', pd.DataFrame, np.ndarray)
def to_str_type(X_df: T) -> T:
    """Converts all columns in the input DataFrame to string type."""
    if not isinstance(X_df, pd.DataFrame) and not isinstance(X_df, np.ndarray):
        # Potremmo provare a convertirlo, ma è meglio assicurarsi che riceva solo i tipi che ci aspettiamo
        raise TypeError(f"[to_str_type] unexpected type: {type(X_df)}")
    return X_df.astype(str)


# Questa funzione è utile per convertire i dati in un formato compatibile con Gradio