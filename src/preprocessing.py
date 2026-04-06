import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class DataPreprocessor:
    @staticmethod
    def clean(df: pd.DataFrame, drop_cols: list = None, filter_values: dict = None) -> pd.DataFrame:
        df = df.copy()
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
            
        for col, values in filter_values.items():
            if col in df.columns:
                df = df[~df[col].isin(values)]
                
        df = df.dropna()
        return df
    
    @staticmethod
    def apply_map(X, map_dict: dict) -> pd.DataFrame:
        return X.apply(lambda col: col.map(map_dict))

    @staticmethod
    def apply_ordinal(X, categories: list[list]) -> pd.DataFrame:
        encoder = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        encoded = encoder.fit_transform(X)

        return pd.DataFrame(encoded, index=X.index, columns=X.columns)

    @staticmethod
    def apply_ohe(X, dtype: str | type = "float32") -> pd.DataFrame:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        encoded = encoder.fit_transform(X)
        encoded_cols = encoder.get_feature_names_out(X.columns)

        return pd.DataFrame(encoded, columns=encoded_cols, index=X.index).astype(dtype)

    @staticmethod
    def cast_numeric(X, numeric_type: str | type = "float32") -> pd.DataFrame:
        return X.apply(pd.to_numeric, errors="coerce").astype(numeric_type)