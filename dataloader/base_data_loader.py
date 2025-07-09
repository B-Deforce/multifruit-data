import json
import os
import sqlite3
import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataloader.config import BDBConfig, BDBDataConfig, IVIAConfig, IVIADataConfig

pd.set_option("future.no_silent_downcasting", True)


@dataclass
class BaseDataLoader:
    """Base class for loading and processing data from a database.
    Args:
        db_path (str): Path to the SQLite database file.
        query_path (str): Path to the SQL query file.
        name (str): Name of the resulting dataset.
        config (Dict): Configuration dictionary containing parameters for data processing.
    """

    db_path: str
    query_path: str
    name: str
    config: tp.Union[BDBDataConfig, IVIADataConfig]

    @classmethod
    def from_config(cls, config_path: str, source: tp.Literal["BDB", "IVIA"]):
        """Create an instance of BaseDataLoader from a configuration file.
        Args:
            config_path (str): Path to the configuration YAML file.
        Returns:
            BaseDataLoader: An instance of BaseDataLoader initialized with parameters from the config file.
        """
        if source == "BDB":
            config = BDBConfig.from_yaml(config_path)
        elif source == "IVIA":
            config = IVIAConfig.from_yaml(config_path)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'BDB' or 'IVIA'.")

        return cls(
            db_path=config.db_path,
            query_path=config.query_path,
            name=config.name,
            config=config.data,
        )

    def load_single_query(self, query_path: str = None, query: str = None):
        """Load a single query from the database.
        Args:
            query_path (str): Path to the SQL query file.
            query (str): SQL query string.
        Returns:
            pd.DataFrame: DataFrame containing the results of the query.
        """
        conn = sqlite3.connect(self.db_path)
        assert query_path or query, "Either query_path or query must be provided"
        if query_path:
            with open(query_path, "r") as sql_file:
                query = sql_file.read()
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def load_query(self):
        """Load a query from the database and return the results as a DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the results of the query.
        """
        conn = sqlite3.connect(self.db_path)
        with open(self.query_path, "r") as sql_file:
            self.df = pd.read_sql_query(sql_file.read(), conn)
            conn.close()
        return self.df

    def save_data(self):
        """Save the processed data to parquet files and scaling parameters to JSON."""
        name = self.name
        os.makedirs(f"data/processed/{name}/", exist_ok=True)
        os.makedirs(f"data/interim/{name}/", exist_ok=True)
        os.makedirs(f"data/interim/{name}/", exist_ok=True)

        total_length = len(self.train_df) + len(self.val_df) + len(self.test_df)
        print(f"Total length of data (train + val + test): {total_length}")

        self.train_df.to_parquet(f"data/processed/{name}/train.parquet", index=False)
        self.val_df.to_parquet(f"data/processed/{name}/val.parquet", index=False)
        self.test_df.to_parquet(f"data/processed/{name}/test.parquet", index=False)
        print(f"Model data saved in data/processed/{name}/")
        with open(f"data/interim/{name}/scaling_params.json", "w") as f:
            json.dump(self.scaling_params, f)
        if self.dummy_map:
            with open(f"data/interim/{name}/dummy_map.json", "w") as f:
                json.dump(self.dummy_map, f)
        print(f"Scaling params saved in data/interim/{name}/scaling_params.json")

    def _fill_missing(self):
        """Fill missing values in the DataFrame based on the configuration."""
        if self.config.fill_missing:
            for col, filler in self.config.fill_missing.items():
                self.df[col] = self.df[col].fillna(filler)

    def _create_group_id(self):
        """Create a composite group ID based on specified columns in the configuration."""
        self.group_id = "composite_group_id"
        self.df[self.group_id] = (
            self.df[self.config.group_id_cols].astype(str).agg("_".join, axis=1)
        )

    def add_time_index(self):
        """Add a time index to the DataFrame based on the group ID."""
        self.df["time_idx"] = self.df.groupby(self.group_id).cumcount()

    def add_dummies(self):
        """Convert specified categorical columns to dummy variables."""
        self.dummy_map = {}
        for col in self.config.dummies:
            mapping = {v: i for i, v in enumerate(self.df[col].unique())}
            self.df[col] = (
                self.df[col].replace(mapping).infer_objects(copy=False).astype("int64")
            )
            self.dummy_map[col] = mapping

    def standardize(self, cols):
        """Standardize specified columns in the DataFrame using min-max scaling.
        Args:
            cols (List[str]): List of column names to standardize.
        Returns:
            None: The DataFrame is modified in place.
        """
        self.scaling_params = {}
        # get idx of cols with nulls
        nan_cols = (
            self.train_df[cols].columns[self.train_df[cols].isna().any()].to_list()
        )
        if nan_cols != []:
            print(f"Warning: null values found for {nan_cols}, computing w/o nulls")
            self.scaling_params["min_nans"] = self.train_df[nan_cols].min(skipna=True)
            self.scaling_params["max_nans"] = self.train_df[nan_cols].max(skipna=True)
            self.train_df = self._minmax_scaler(
                self.train_df, nan_cols, self.scaling_params, nan_mode=True
            )
            self.val_df = self._minmax_scaler(
                self.val_df, nan_cols, self.scaling_params, nan_mode=True
            )
            self.test_df = self._minmax_scaler(
                self.test_df, nan_cols, self.scaling_params, nan_mode=True
            )
        # for remaining cols
        cols = [col for col in cols if col not in nan_cols]
        self.scaling_params["min"] = self.train_df[cols].min()
        self.scaling_params["max"] = self.train_df[cols].max()
        self.train_df = self._minmax_scaler(self.train_df, cols, self.scaling_params)
        self.val_df = self._minmax_scaler(self.val_df, cols, self.scaling_params)
        self.test_df = self._minmax_scaler(self.test_df, cols, self.scaling_params)
        for item in self.scaling_params:
            self.scaling_params[item] = self.scaling_params[item].to_dict()

    def _minmax_scaler(self, df, cols, scaling_params, nan_mode=False):
        """Apply min-max scaling to specified columns in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame to scale.
            cols (List[str]): List of column names to scale.
            scaling_params (Dict): Dictionary containing min and max values for scaling.
            nan_mode (bool): Whether to handle NaN values in the scaling.
        Returns:
            pd.DataFrame: DataFrame with scaled columns.
        """
        df[cols] = (
            2
            * (
                (df[cols] - scaling_params["min_nans" if nan_mode else "min"])
                / (
                    scaling_params["max_nans" if nan_mode else "max"]
                    - scaling_params["min_nans" if nan_mode else "min"]
                )
            )
            - 1
        )
        return df

    def train_val_test_split(self, date_col, train_cutoff, target_col, field_col):
        """Split the DataFrame into training, validation, and test sets based on a date column.
        Args:
            date_col (str): Name of the date column to use for splitting.
            train_cutoff (str): Cutoff date for the training set.
            target_col (str): Name of the target column.
            field_col (Optional[str]): Name of the field column to use for splitting.
        Returns:
            None: The DataFrame is split into train_df, val_df, and test_df attributes.
        """
        df = self.df.copy()

        if field_col:
            print("Splitting by field_id")
            self.train_df = df[
                df[field_col].astype(str) != str(self.config.field_id)
            ].copy()
            # assign half of the field data to val and test randomly, by group
            valtest_df = df[
                df[field_col].astype(str) == str(self.config.field_id)
            ].copy()
            # Get unique group_ids
            unique_groups = valtest_df[self.group_id].unique()

            # Shuffle the unique group_ids
            shuffled_groups = (
                pd.Series(unique_groups).sample(frac=1, random_state=42).tolist()
            )

            # Split the group_ids into two halves
            split = int(len(shuffled_groups) / 2)
            val_groups = shuffled_groups[:split]
            test_groups = shuffled_groups[split:]

            # Create validation and test DataFrames based on the group splits
            self.val_df = valtest_df[valtest_df[self.group_id].isin(val_groups)].copy()
            self.test_df = valtest_df[
                valtest_df[self.group_id].isin(test_groups)
            ].copy()

            self.val_df = self.val_df.sort_values(
                by=[self.group_id, date_col]
            ).reset_index(drop=True)
            self.test_df = self.test_df.sort_values(
                by=[self.group_id, date_col]
            ).reset_index(drop=True)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
            self.train_df = df[df[date_col] <= train_cutoff].copy()

            valtest_df = df[df[date_col] > train_cutoff]
            self.val_df = valtest_df.copy()
            self.test_df = valtest_df.copy()

            self.val_df.loc[self.val_df[date_col].dt.day > 14, target_col] = np.nan
            self.test_df.loc[self.test_df[date_col].dt.day <= 14, target_col] = np.nan

            self.val_df = self.val_df.sort_values(
                by=[self.group_id, date_col]
            ).reset_index(drop=True)
            self.test_df = self.test_df.sort_values(
                by=[self.group_id, date_col]
            ).reset_index(drop=True)
