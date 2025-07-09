import os
import typing as tp
from dataclasses import dataclass

import yaml
from beartype import beartype


@beartype
@dataclass
class VGConfig:
    """Configuration for Van Genuchten parameters.
    Args:
        alpha: Alpha parameter for the Van Genuchten model.
        n: N parameter for the Van Genuchten model.
        theta_s: Saturated water content.
        theta_r: Residual water content.
    """

    alpha: float
    n: float
    theta_s: float
    theta_r: float


@beartype
@dataclass
class BDBDataConfig:
    """Configuration for BDB data loading.
    Args:
        group_id_cols: List of columns used for grouping data.
        pivot: Boolean indicating whether to pivot the data (e.g., sensor depth from long to wide).
        train_cutoff: Date string indicating the cutoff for training data.
        date_col: Name of the column containing date information.
        target_col: Name of the target variable column.
        standardize_cols: List of columns to standardize.
        field_col: Optional name of the column containing field ID.
        dummies: Optional list of columns to convert to dummy variables.
        fill_missing: Dictionary mapping column names to values used for filling missing data.
        vg_parameters: Dictionary containing Van Genuchten parameters for soil moisture conversion,
            for a specific orchard.
    """

    group_id_cols: list[str]
    pivot: bool
    train_cutoff: str
    date_col: str
    target_col: str
    standardize_cols: list[str]
    field_col: str | None = None
    dummies: list[str] | None = None
    fill_missing: dict[str, str | int | float] | None = None
    vg_parameters: dict[str, VGConfig] | None = None


@beartype
@dataclass
class IVIADataConfig:
    """Configuration for IVIA data loading.
    Args:
        group_id_cols: List of columns used for grouping data.
        train_cutoff: Date string indicating the cutoff for training data.
        date_col: Name of the column containing date information.
        target_col: Name of the target variable column.
        standardize_cols: List of columns to standardize.
        use_meteo: String indicating which meteorological data to use. The database contains meteo data
            at daily and half-hourly intervals. The "daily" option uses daily meteo data, while the "hh"
            option uses half-hourly meteo data, which can be aggregated to the sample_rate.
        vwc_chul: Dictionary containing SQL query path for VWC data from Chullila.
        vwc_lir: Dictionary containing SQL query paths for VWC data from Liria.
        tree: Dictionary containing SQL query path for tree data.
        irrigation: Dictionary containing SQL query path for irrigation data.
        daily_meteo: Dictionary containing SQL query path for daily meteorological data.
        hh_meteo: Dictionary containing SQL query path for half-hourly meteorological data.
        swp: Dictionary containing SQL query path for soil water potential data.
        dummies: Optional list of columns to convert to dummy variables.
        field_col: Optional name of the column containing field ID.
        vwc_aggregation_scheme: Optional dictionary specifying how to aggregate VWC data.
            This is a dictionary where keys are aggregation methods (e.g., "mean") and values
            are lists of column names to aggregate. If None, no aggregation is performed.
        sample_rate: String indicating the sample rate (e.g., "4h"). By default, the data is sampled
            at half hourly intervals. The sample_rate can be used to aggregate the data to a different frequency.
            We have found that 4h is a good sample rate for the IVIA data.
        window_size: String indicating the window size for rolling calculations (e.g., "120h").
            This is expressed in units of `sample_rate`. Hence, if the sample_rate is "4h",
            a window_size of "120h" means 30 time-steps, totalling 5 days.
            We have found that 120h is a good window size for the IVIA data.
        fill_missing: Dictionary mapping column names to values used for filling missing data.
        map_to_bdb: Boolean indicating whether to map the IVIA data to the BDB format.
            If True, the data will be transformed to match the BDB format (e.g. column names),
            which is useful for merging both datasets into a single dataset to train one single model.
        bdb_path_train: Optional path to the BDB training data file. Required if `map_to_bdb` is True.
        bdb_path_val: Optional path to the BDB validation data file. Required if `map_to_bdb` is True.
        bdb_path_test: Optional path to the BDB test data file. Required if `map_to_bdb` is True.
    """

    group_id_cols: list[str]
    train_cutoff: str
    date_col: str
    target_col: str
    standardize_cols: list[str]
    use_meteo: tp.Literal["daily", "hh"]
    vwc_chul: dict[str, str]
    vwc_lir: dict[str, list[str]]
    tree: dict[str, str]
    irrigation: dict[str, str]
    daily_meteo: dict[str, str]
    hh_meteo: dict[str, str]
    swp: dict[str, str]
    dummies: list[str] | None = None
    field_col: str | None = None
    vwc_aggregation_scheme: dict[str, list[str]] | None = None
    sample_rate: str = "4h"
    window_size: str = "120h"
    fill_missing: dict[str, str | int | float] | None = None
    map_to_bdb: bool = False
    bdb_path_train: str | None = None
    bdb_path_val: str | None = None
    bdb_path_test: str | None = None

    def __post_init__(self):
        if self.map_to_bdb:
            if not all([self.bdb_path_train, self.bdb_path_val, self.bdb_path_test]):
                raise ValueError(
                    "If map_to_bdb is True, bdb_path_train, bdb_path_val, and bdb_path_test must be provided."
                )


@beartype
@dataclass
class BaseConfig:
    """Base configuration class for data loaders.
    Args:
        name: Name of the configuration.
        db_path: Path to the database file.
        query_path: Path to the SQL query file.
    """

    name: str
    db_path: str
    query_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg_dict = raw.get("parameters", raw)
        return cls.from_dict(cfg_dict)

    def __post_init__(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")


@beartype
@dataclass(kw_only=True)
class BDBConfig(BaseConfig):
    """Configuration for BDB data loader.
    Args:
        data: BDBDataConfig instance containing data loading configurations.
    """

    data: BDBDataConfig

    @classmethod
    def from_dict(cls, d: dict):
        if d["data"].get("vg_parameters"):
            d["data"]["vg_parameters"] = {
                k: VGConfig(**v) for k, v in d["data"]["vg_parameters"].items()
            }
        d["data"] = BDBDataConfig(**d["data"])
        return cls(**d)


@beartype
@dataclass(kw_only=True)
class IVIAConfig(BaseConfig):
    """Configuration for IVIA data loader.
    Args:
        data: IVIADataConfig instance containing data loading configurations.
    """

    data: IVIADataConfig

    @classmethod
    def from_dict(cls, d: dict):
        d["data"] = IVIADataConfig(**d["data"])
        return cls(**d)
