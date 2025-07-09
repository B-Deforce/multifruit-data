# Multi-Fruit Datasets

This repository contains preprocessing code for various datasets related to fruit orchards.
This repository currently include code for the following datasets:
- BDB (Bodemkundige Dienst van België/Soil Service of Belgium) dataset
- IVIA (Institut Valencià d'Investigacions Agràries) dataset

In this repository, we provide code to preprocess the datasets and convert them into a format suitable for machine learning tasks.

The code is organized into the following directories:
- `dataloader/`: Contains the data loading and preprocessing code. It includes the base data loader and specific loaders for BDB and IVIA datasets. It also includes data schemas in the `config.py` file.
- `config/`: Contains configuration files for the datasets, including database paths, queries, and data parameters (e.g., where to cut-off training data, how to fill missing values, ...)
- `queries/`: Contains SQL queries for loading data from the databases.
- `scripts/`: Contains shell scripts for running the data loading and preprocessing code.
- `main.py`: The main entry point for running the data loading and preprocessing code.

## Config details
The configuration files in the `config/` directory define the parameters for loading and preprocessing the datasets. The configuration files are structured as follows:
- `bdb/bdb.yml`: Configuration file for the standard BDB dataset.
- `bdb/bdb_swp_to_vwc.yml`: Configuration file for the BDB dataset with soil water potential (SWP) converted to volumetric water content (VWC), using the Van Genuchten model.
- `ivia/ivia.yml`: Configuration file for the standard IVIA dataset.

Note, complete schemas for the config files can be found in the `dataloader/config.py` file and are enforced at runtime.

## Usage
To run the data loading and preprocessing code, you can use the `load_data.sh` script. The script takes a configuration file as an argument. For example, to run the BDB dataset preprocessing with SWP to VWC conversion, you can set the name in the shell script as `bdb/bdb_swp_to_vwc`, define the dataloader type as `bdb`, and run the following command from root:

```bash
sh scripts/load_data.sh
```
This will execute the `main.py` script with the specified configuration file, loading and preprocessing the data according to the parameters defined in the configuration file.

The script will output the processed data to the `data/` directory. The output consists of a folder (following the name defined in the config file) in `interim/` containing a mapping of the dummy variables and a scaling file, containing the scaling parameters used for standardization. Additionally, it will create a folder in the `processed/` folder containing the train-test-val split.

## TO DO
- Integrate volumetric water content (VWC) for Liria directly in IVIA database. As these files were delivered later, they are currently included as `.csv`. files. However, this data *is* present in the final datasets. It would just be more convenient to have it in the database.
- Improve the loading of the IVIA dataset, currently it loads many separate files, which is not very convenient. It would be better to load the data from the database directly using SQL queries.