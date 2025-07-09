import argparse
import typing as tp

from beartype import beartype

from dataloader.bdb_data_loader import BDB_DataLoader
from dataloader.ivia_data_loader import IVIA_DataLoader


@beartype
def get_loader(loader_type: tp.Literal["bdb", "ivia"]):
    if loader_type == "bdb":
        return BDB_DataLoader
    elif loader_type == "ivia":
        return IVIA_DataLoader
    else:
        raise ValueError(f"Unknown loader type: {loader_type}. Choose 'bdb' or 'ivia'.")


def load_data(config_name, loader_type: tp.Literal["bdb", "ivia"]):
    loader = get_loader(loader_type)
    loader = loader.from_config(f"config/{config_name}.yml")
    loader.load_and_process_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument(
        "--dataloader",
        type=str,
        required=True,
        choices=["bdb", "ivia"],
    )
    args = parser.parse_args()
    load_data(args.config_name, args.dataloader)
