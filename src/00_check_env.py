import importlib.util
from pathlib import Path

import torch


DATASET_PATHS = {
    "womd": "/mnt/data1/wzy/datasets/waymo_open_motion",
    "occ3d_waymo": "/mnt/data1/wzy/datasets/occ3d_waymo",
    "occ3d_nuscenes": "/mnt/data1/wzy/datasets/occ3d_nuscenes",
    "nuscenes": "/mnt/data1/wzy/datasets/nuscenes",
    "nuplan": "/mnt/data1/wzy/datasets/nuplan",
}


PACKAGES = [
    "torch",
    "tensorflow",
    "waymo_open_dataset",
    "numpy",
    "shapely",
    "yaml",
]


def package_exists(name):
    return importlib.util.find_spec(name) is not None


def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"cuda:{index}", torch.cuda.get_device_name(index))

    print("\nPackages:")
    for name in PACKAGES:
        print(f"{name}: {package_exists(name)}")

    print("\nDataset paths:")
    for name, path in DATASET_PATHS.items():
        p = Path(path)
        print(f"{name}: {p.exists()}  {path}")


if __name__ == "__main__":
    main()
