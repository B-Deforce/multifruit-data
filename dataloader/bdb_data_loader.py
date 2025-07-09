from dataclasses import dataclass

import pandas as pd

from dataloader.base_data_loader import BaseDataLoader


@dataclass
class BDB_DataLoader(BaseDataLoader):
    """Data loader for the BDB dataset.
    This class extends the BaseDataLoader to handle specific data processing
    tasks for the BDB dataset, including pivoting by depth, converting SWP to VWC,
    and dropping rogue sensors.
    """

    @classmethod
    def from_config(cls, config_path: str):
        return super().from_config(config_path, source="BDB")

    dummy_map = None

    def load_and_process_data(self):
        self.load_query()
        self._fill_missing()
        self._create_group_id()
        self._drop_rogue_sensors()
        if self.config.pivot:
            print("Pivoting...")
            self.pivot_by_depth()
        if self.config.dummies:
            print("Adding dummies...")
            self.add_dummies()
        print("Splitting data...")
        self.add_month()
        self.add_time_index()
        if self.config.vg_parameters:
            print("Converting SWP to VWC...")
            for moist in ["avg_moist_30", "avg_moist_60", "avg_moist_90"]:
                self.df[moist] = self.df.apply(
                    lambda row: self.convert_swp_to_vwc(
                        row[moist], row["orchard_name"]
                    ),
                    axis=1,
                )
        self.train_val_test_split(
            self.config.date_col,
            self.config.train_cutoff,
            self.config.target_col,
            self.config.field_col,
        )
        print("Standardizing data...")
        self.standardize(self.config.standardize_cols)
        self.save_data()

    def _drop_rogue_sensors(self):
        """Custom function for BDB datasets.
        While hard-coding the sensor_id and measurement_year is not ideal,
        it's OK as it's a dataset specific class.
        This function drops sensors that have rogue values, NaN values,
        fewer than 20 measurements, or non-consecutive dates.
        """
        # manual sensor drop
        # self.df = self.df[
        # ~((self.df["sensor_id"] == 67) & (self.df["measurement_year"] == "2009"))
        # ].copy()

        all_sensors = self.df[self.group_id].unique()

        # Identify rogue sensors: sensors that hit the lower limit of the moisture sensor
        to_drop_rogue = self.df[self.df["avg_moist"] <= -199.0][self.group_id].unique()
        self.df = self.df[~self.df[self.group_id].isin(to_drop_rogue)].copy()
        # Identify sensors with all NaN values
        all_nan = self.df.groupby(self.group_id)["avg_moist"].apply(
            lambda x: x.isnull().all()
        )
        # Get the list of groups to drop
        to_drop_nan = all_nan[all_nan].index
        # Drop those groups from the DataFrame
        self.df = self.df[~self.df[self.group_id].isin(to_drop_nan)]

        # sensors with fewer than 20 measurements
        sensor_counts = self.df[self.group_id].value_counts()
        to_drop_few = sensor_counts[sensor_counts < 20].index
        self.df = self.df[~self.df[self.group_id].isin(to_drop_few)]

        # sensors with non-consecutive dates
        to_drop_dates = self._check_dates()
        self.df = self.df[~self.df[self.group_id].isin(to_drop_dates)]

        to_drop_all = (
            len(to_drop_rogue)
            + len(to_drop_nan)
            + len(to_drop_few)
            + len(to_drop_dates)
        )
        # Print the results
        print(
            f"Bad sequences: {to_drop_all}, dropping them out of {len(all_sensors)} sequences."
        )
        print(
            f"{len(to_drop_rogue)} of those hit at least once -199.0 value, indicating the measurement threshold."
        )
        print(f"{len(to_drop_nan)} of those have only NaN values.")
        print(f"{len(to_drop_few)} of those have fewer than 20 measurements.")
        print(f"{len(to_drop_dates)} of those have non-consecutive dates.")

    def pivot_by_depth(self):
        """This function pivots the DataFrame by depth.
        It transforms the DataFrame from a long format to a wide format,
        where each depth becomes a separate column for both average moisture and soil temperature.
        The original DataFrame has columns like 'avg_moist' and 'avg_soil_temp' for each depth.
        By pivoting on depth, we go from 'MOIST | DEPTH' to 'MOIST_DEPTH1 | MOIST_DEPTH2 | ...'
        """
        pivot_df = self.df.pivot_table(
            index=["orchard_id", "plot_id", "measurement_date", "measurement_year"],
            columns="depth_cm",
            values=["avg_moist", "avg_soil_temp"],
        ).reset_index()
        pivot_df.columns = [
            "_".join(map(str, col)).strip("_") for col in pivot_df.columns.values
        ]
        pivot_df[self.group_id] = (
            pivot_df["orchard_id"].astype(str)
            + "_"
            + pivot_df["plot_id"].astype(str)
            + "_"
            + pivot_df["measurement_year"].astype(str)
        )
        # keep only temp at depth 30 as there is almost no difference compared to 60 and 90
        pivot_df = pivot_df.drop(columns=["avg_soil_temp_60", "avg_soil_temp_90"])
        pivot_df = self._fill_missings_after_pivot(pivot_df)

        self.df = self.df.drop(
            columns=[
                "depth_cm",
                "avg_soil_temp",
                "avg_moist",
                self.group_id,
                "measurement_year",
            ]
        ).drop_duplicates()
        self.df = pivot_df.merge(
            self.df,
            on=["orchard_id", "plot_id", "measurement_date"],
            how="left",
        )

    def add_month(self):
        """Add a month column to the DataFrame based on the measurement_date."""
        self.df["month"] = pd.to_datetime(self.df["measurement_date"]).dt.month

    def _fill_missings_after_pivot(self, df):
        """Fill missing values in the pivoted DataFrame.
        This function fills missing values for average moisture and soil temperature
        columns after pivoting. It also creates missing value indicators for average moisture.
        If all values in a group are missing, it drops that group.
        """
        for col in df.columns:
            if "avg_moist" in col:
                df[col + "_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(1)
            if "avg_soil_temp" in col:
                df[col] = df[col].fillna(-10)
            elif df[col].isna().sum() > 0:
                raise ValueError(
                    f"Missing values in the pivoted DataFrame after filling missing values for {col}"
                )
        missing_cols = [col for col in df.columns if "missing" in col]
        # if all are missing, drop the group
        all_missing = df[missing_cols].sum(axis=1)
        drop_groups = df[all_missing == len(missing_cols)][self.group_id].unique()
        print(
            f"Dropping {len(drop_groups)} groups with all missing values after pivoting"
        )
        df = df[~df[self.group_id].isin(drop_groups)].copy()
        return df

    def _check_dates(self):
        """Check for non-consecutive dates in the DataFrame.
        This function groups the DataFrame by the group_id and checks if the measurement dates
        within each group are consecutive. It returns a list of group_ids that have non-consecutive dates.
        """
        non_consecutive_groups = self.df.groupby(self.group_id).apply(
            lambda x: (
                any(
                    pd.to_datetime(x["measurement_date"]).diff().dropna()
                    != pd.Timedelta(days=1)
                )
            )
        )
        # Extract group_ids with non-consecutive dates
        non_consecutive_group_ids = non_consecutive_groups[
            non_consecutive_groups
        ].index.tolist()
        return non_consecutive_group_ids

    def convert_swp_to_vwc(self, value, crop_name):
        """Convert SWP to VWC using van Genuchten model."""
        value = self._kpa_to_cm_H2O(value)
        value_converted = self._van_genuchten(
            value,
            alpha=self.config.vg_parameters[crop_name].alpha,
            n=self.config.vg_parameters[crop_name].n,
            m=1 - 1 / self.config.vg_parameters[crop_name].n,
            theta_r=self.config.vg_parameters[crop_name].theta_r,
            theta_s=self.config.vg_parameters[crop_name].theta_s,
        )
        return value_converted

    def _van_genuchten(self, psi, alpha, n, m, theta_r, theta_s):
        return theta_r + (theta_s - theta_r) / ((1 + (alpha * abs(psi)) ** n) ** m)

    def _kpa_to_cm_H2O(self, kpa):
        return kpa * 10.197
