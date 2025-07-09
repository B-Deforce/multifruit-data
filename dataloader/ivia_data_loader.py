from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataloader.base_data_loader import BaseDataLoader


@dataclass
class IVIA_DataLoader(BaseDataLoader):
    """Data loader for the IVIA dataset.
    This class extends the BaseDataLoader to handle specific data processing
    tasks for the IVIA dataset, including loading and merging VWC, meteo, irrigation,
    and SWP data, filtering dates, creating group IDs, and mapping to the BDB structure.
    """

    @classmethod
    def from_config(cls, config_path: str):
        return super().from_config(config_path, source="IVIA")

    def load_and_process_data(self):
        """Load and process the IVIA dataset."""
        vwc_meteo = self.load_vwc_and_merge_with_meteo()
        irr = self._load_irr_df()
        vwc_meteo_irr = self.merge_with_irr(vwc_meteo, irr)
        swp = self._load_swp_df()
        self.df = self.merge_with_swp(vwc_meteo_irr, swp)
        self.df = self.filter_dates(self.df)
        self._create_group_id()
        self.add_time_index()
        if self.config.dummies:
            print("Adding dummies...")
            self.add_dummies()
        self.train_val_test_split(
            self.config.date_col,
            self.config.train_cutoff,
            self.config.target_col,
            self.config.field_col,
        )
        print("Standardizing data...")
        self.standardize(self.config.standardize_cols)
        if self.config.map_to_bdb:
            print("Mapping to bdb...")
            self.map_to_bdb()
        self.save_data()

    def _load_vwc_chul(self):
        """Load VWC data from the Chullila dataset."""
        q = self.config.vwc_chul["query_path"]
        df = self.load_single_query(q)
        df["date"] = pd.to_datetime(df["datetime"]).dt.date.astype(str)
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.round("30min").astype(str)
        # whenever tree_id equals 3_2_0, add 'treatment" column with value 1, 3 otherwise
        df["treatment"] = np.where(df["tree_id"] == "3_2_0", 1, 3)  # temp solution
        df["treatment"] = df["treatment"].astype(int).astype(str)
        df["field_id"] = (
            df["tree_id"].str.split("_", expand=True).iloc[:, -1]
        )  # str by default
        df.drop(columns=["id"], inplace=True)
        return df

    def _load_vwc_lir(self):
        """Load VWC data from the Liria dataset."""
        q = self.config.vwc_lir["query_path"]
        df = [pd.read_csv(f) for f in q]
        df = pd.concat(df, ignore_index=True, axis=0)
        df["date"] = pd.to_datetime(df["datetime"]).dt.date.astype(str)
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.round("30min").astype(str)
        df["treatment"] = df["treatment"].astype(int).astype(str)
        df["field_id"] = (
            df["tree_id"].str.split("_", expand=True).iloc[:, -1]
        )  # str by default
        return df

    def _load_all_vwc(self):
        """Load all VWC data from both Liria and Chullila datasets."""
        vwc_lir = self._load_vwc_lir()
        vwc_chul = self._load_vwc_chul()
        return pd.concat([vwc_lir, vwc_chul], ignore_index=True, axis=0)

    def _load_daily_meteo(self):
        """Load daily meteorological data."""
        q = self.config.daily_meteo["query_path"]
        df = self.load_single_query(q)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        for col in ["date", "field_id"]:
            df[col] = df[col].astype(str)
        df = df[["date", "field_id", "temp", "precip_mm", "eto_mm"]]
        return df

    def _load_hh_meteo(self):
        """Load half-hourly meteorological data."""
        q = self.config.hh_meteo["query_path"]
        df = self.load_single_query(q)
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed").dt.floor("s")
        df.loc[df["id"] == 112546, "datetime"] -= pd.Timedelta(
            days=1
        )  # one data point is off by a day
        for col in ["datetime", "field_id"]:
            df[col] = df[col].astype(str)
        df.drop(columns=["id", "date"], inplace=True)
        return df

    def _load_tree_df(self):
        """Load tree data."""
        q = self.config.tree["query_path"]
        df = self.load_single_query(q)
        df["plot_id"] = df["plot_id"].astype(str)
        return df

    def _load_swp_df(self):
        """Load soil water potential (SWP) data."""
        q = self.config.swp["query_path"]
        df = self.load_single_query(q)
        df.drop_duplicates(inplace=True)
        df["datetime"] = (
            pd.to_datetime(df["date"]) + pd.to_timedelta(12, unit="h")
        ).astype(str)
        df["treatment"] = df["treatment"].astype(str)
        return df

    def _load_irr_df(self):
        """Load irrigation data."""
        q = self.config.irrigation["query_path"]
        df = pd.read_csv(q)
        for col in ["plot_id", "field_id", "treatment", "date"]:
            df[col] = df[col].astype(str)
        return df

    def load_vwc_and_merge_with_meteo(self):
        """Load VWC data and merge it with meteorological data."""
        vwc = self._load_all_vwc()
        trees = self._load_tree_df()
        vwc = vwc.merge(trees, on="tree_id", how="left")
        if self.config.use_meteo == "hh":
            meteo = self._load_hh_meteo()
            merged_df = vwc.merge(meteo, on=["datetime", "field_id"], how="left")
            merged_df = self._aggregate_vwc_and_hh_meteo(merged_df)
        else:
            meteo = self._load_daily_meteo()
            vwc.drop(columns=["datetime"], inplace=True)
            merged_df = vwc.merge(meteo, on=["date", "field_id"], how="left")
            merged_df = self._aggregate_vwc_and_daily_meteo(merged_df)

        return merged_df

    def merge_with_irr(self, vwc_meteo, irr):
        """Merges the vwc_meteo dataframe with the irrigation dataframe
        on:
            - date (or datetime if using half-hourly meteo)
            - plot_id
            - treatment
            - field_id
        This function fills missing irrigation values with 0.
        It also checks for duplicates in the merged dataframe based on tree_id, date (or datetime), and treatment.
        """
        vwc_meteo_irr = vwc_meteo.merge(
            irr, on=["date", "plot_id", "treatment", "field_id"], how="left"
        )
        vwc_meteo_irr["irr_mm"] = vwc_meteo_irr["irr_mm"].fillna(0)
        dups = (
            ["tree_id", "datetime", "treatment"]
            if self.config.use_meteo == "hh"
            else ["tree_id", "date", "treatment"]
        )
        assert vwc_meteo_irr[dups].duplicated().sum() == 0
        return vwc_meteo_irr

    def merge_with_swp(self, vwc_meteo_irr, swp):
        """Merges the vwc_meteo_irr dataframe with the swp dataframe
        on:
            - datetime (as swp is measured at noon, we add 12 hours to the date column to match
            the datetime column in vwc_meteo_irr)
            - tree_id
            - treatment
        """
        if self.config.use_meteo == "hh":
            merged_df = vwc_meteo_irr.merge(
                swp.drop(columns=["date"]),
                on=["datetime", "tree_id", "treatment"],
                how="left",
            )
            merged_df = merged_df.dropna(
                subset=["avg_temp", "avg_humidity"]
            )  # no weather data
        else:
            merged_df = vwc_meteo_irr.merge(
                swp.drop(columns=["datetime"]),
                on=["date", "tree_id", "treatment"],
                how="left",
            )
        ext_swp = (merged_df["avg_swp"] < -2.0).sum()
        print(f"SWP data with avg_swp < -2.0: {ext_swp} rows. Dropped.")
        merged_df["avg_swp"] = merged_df["avg_swp"].apply(
            lambda x: np.nan if x < -2.0 else x
        )
        return merged_df

    def _aggregate_vwc_and_hh_meteo(self, df):
        """Aggregate VWC and half-hourly meteorological data to the sample rate defined in the config."""
        agg_scheme = {}
        config = self.config.vwc_aggregation_scheme
        for agg_type, columns in config.items():
            for column in columns:
                agg_scheme[column] = agg_type
        sr = self.config.sample_rate
        print(f"Aggregating to sample rate: {sr}...")
        df = (
            df.assign(datetime=lambda df: pd.to_datetime(df["datetime"]))
            .groupby("tree_id")
            .apply(lambda g: g.resample(sr, on="datetime").aggregate(agg_scheme))
        )
        df.drop(columns=["tree_id"], inplace=True)  # since it's already in index
        df.reset_index(inplace=True)
        df["datetime"] = df["datetime"].astype(
            str
        )  # convert back to string for merging
        return df

    def _aggregate_vwc_and_daily_meteo(self, df):
        """Aggregate VWC and daily meteorological data to day level."""
        print(
            "Aggregating to day level since we're using daily meteo data, ignoring sample_rate and aggregation_scheme..."
        )
        obj_cols = df.select_dtypes(include=["object"]).columns
        cont_cols = df.select_dtypes(exclude=["object"]).columns
        df = df.groupby(["tree_id", "date"]).agg(
            {
                **{col: "last" for col in obj_cols},
                **{col: "mean" for col in cont_cols},
            }
        )
        df.drop(columns=["date", "tree_id"], inplace=True)
        df.reset_index(inplace=True)
        return df

    def filter_dates(self, df):
        """Filter the DataFrame to only include data between the start and end dates of SWP measurements."""
        date_col = self.config.date_col
        df[date_col] = pd.to_datetime(df[date_col])
        df["month"] = df[date_col].dt.month
        df["year"] = df[date_col].dt.year
        if self.config.use_meteo == "hh":
            df["hour"] = df[date_col].dt.hour
        swp_start_date = df.dropna(subset=["avg_swp"])[date_col].min()
        swp_start_date = swp_start_date - pd.Timedelta(self.config.window_size)
        swp_end_date = df.dropna(subset=["avg_swp"])[date_col].max()
        df = self._get_data_btwn_dates(df, date_col, swp_start_date, swp_end_date)
        return df

    def _get_data_btwn_dates(self, df, date_col, start_date, end_date):
        return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    def add_time_idx(df, date_col, group_col):
        df = df.sort_values(by=[i for i in group_col] + [date_col])
        df["time_idx"] = df.groupby(by=group_col).cumcount()
        return df

    def map_to_bdb(self):
        """Map the IVIA data to the BDB data structure.
        This is needed when e.g. training one model for all crops.
        """
        mapper = {
            "vmc_10": "avg_moist_30",
            "vmc_30": "avg_moist_60",
            "treatment": "irrigation_treatment",
            "eto_mm": "ETo",
            "precip_mm": "precip_daily",
            "irr_mm": "irrigation_amount",
            "avg_swp": "swp_mpa",
        }
        for attr in ["train_df", "val_df", "test_df", "df"]:
            df = getattr(self, attr)
            if df is not None:  # Ensure df is not None
                df = df.rename(columns=mapper)
                df["avg_moist_30_missing"] = df["avg_moist_30"].isna().astype(int)
                df["avg_moist_60_missing"] = df["avg_moist_60"].isna().astype(int)
                setattr(self, attr, df)
        self.scaling_params["min_nans"]["swp_mpa"] = self.scaling_params[
            "min_nans"
        ].pop("avg_swp")
        self.scaling_params["max_nans"]["swp_mpa"] = self.scaling_params[
            "max_nans"
        ].pop("avg_swp")
        self._concat_to_bdb(mapper)

    def _concat_to_bdb(self, mapper):
        """Concatenate the BDB data to the IVIA data.
        This is done to create a unified dataset for training.
        It loads the BDB data, renames columns according to the mapper,
        and concatenates it with the IVIA data.
        """
        print("Concatenating bdb to IVIA")
        # load bdb
        bdb_cols = list(mapper.values())
        bdb_cols += [
            "avg_moist_30_missing",
            "avg_moist_60_missing",
            "composite_group_id",
            "time_idx",
        ]
        bdb_df_train = pd.read_parquet(self.config.bdb_path_train)
        bdb_df_val = pd.read_parquet(self.config.bdb_path_val)
        bdb_df_test = pd.read_parquet(self.config.bdb_path_test)
        bdb_df_train = bdb_df_train[bdb_cols]
        bdb_df_val = bdb_df_val[bdb_cols]
        bdb_df_test = bdb_df_test[bdb_cols]
        for df in [bdb_df_train, bdb_df_val, bdb_df_test]:
            df["croptype"] = 0
            df["composite_group_id"] = df["composite_group_id"] + 1000
        for df in [self.train_df, self.val_df, self.test_df]:
            df["croptype"] = 1

        # concat
        self.train_df = pd.concat([self.train_df, bdb_df_train], axis=0)
        self.val_df = pd.concat([self.val_df, bdb_df_val], axis=0)
        self.test_df = pd.concat([self.test_df, bdb_df_test], axis=0)
