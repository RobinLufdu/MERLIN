from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import torch
from scipy.io import loadmat
import h5py
import os

class MatlabFileReader:
    def __init__(self, file_path: Union[str, Path], device: str = "cpu", return_tensor: bool = False):
        self.file_path = str(file_path)
        self.device = device
        self.return_tensor = return_tensor
        self._load_file_()

    def _load_file_(self):
        if not Path(self.file_path).exists():
            raise FileNotFoundError(self.file_path)
        try:
            self.mat_data = loadmat(self.file_path)  # not for v7.3
            self._backend = "scipy"
            self.keys = [k for k in self.mat_data.keys() if not k.startswith("__")]
        except NotImplementedError:
            # Likely v7.3 HDF5
            self._backend = "h5py"
            self.h5 = h5py.File(self.file_path, "r")
            self.keys = list(self.h5.keys())

    def list_variables(self) -> List[str]:
        return self.keys

    def read_file(self, section: str) -> Union[np.ndarray, torch.Tensor]:
        if section not in self.keys:
            raise ValueError(f"Invalid section '{section}'. Available: {self.keys}")
        if self._backend == "scipy":
            arr = self.mat_data[section]
        else:
            arr = np.array(self.h5[section])

        if arr.dtype == object:
            raise TypeError(f"Section '{section}' is non-numeric (object dtype). Not supported.")
        if self.return_tensor:
            return torch.as_tensor(arr).to(self.device)
        return arr

    def __repr__(self):
        return (f"MatlabFileReader(file_path={self.file_path}, device={self.device}, "
                f"return_tensor={self.return_tensor}, backend={getattr(self, '_backend', '?')})")
    

def load_matrix_from_path(
    path: str,
    key: str | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    must_be_2d: bool = True,
) -> torch.Tensor:
    """
    Load a matrix/tensor from a given file path.

    Supported formats:
      - .pt/.pth/.ckpt: torch.save outputs (either a bare tensor or a dict of tensors)
      - .npy/.npz     : NumPy arrays (optionally keyed for .npz)
      - .mat          : MATLAB file (requires scipy)
      - .txt/.csv     : Plain text numeric matrix (parsed via numpy)

    Args:
      path: file path on disk
      key:  optional key if the file stores a dict/npz/mat
      device: target device for the returned tensor
      dtype: target dtype for the returned tensor
      must_be_2d: if True, assert the loaded tensor is 2D

    Returns:
      torch.Tensor on the requested device/dtype
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    tensor = None

    if ext in [".pt", ".pth", ".ckpt"]:
        # Load into CPU first (safe), then move/cast.
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            tensor = obj
        elif isinstance(obj, dict):
            # If key is not provided, try to auto-pick when there is exactly one tensor-like entry
            if key is None:
                tensorish = {k: v for k, v in obj.items() if torch.is_tensor(v)}
                if len(tensorish) == 1:
                    key = next(iter(tensorish))
                else:
                    raise KeyError(
                        f"File contains multiple entries ({list(obj.keys())}); please provide key="
                    )
            if key not in obj:
                raise KeyError(f"Key '{key}' not found. Available keys: {list(obj.keys())}")
            t = obj[key]
            if not torch.is_tensor(t):
                raise TypeError(f"Entry '{key}' is not a tensor (type={type(t)})")
            tensor = t
        else:
            raise TypeError(f"Unsupported object type in {ext}: {type(obj)}")

    elif ext == ".npy":
        arr = np.load(path)
        tensor = torch.from_numpy(arr)

    elif ext == ".npz":
        npz = np.load(path)
        if key is None:
            if len(npz.files) == 1:
                key = npz.files[0]
            else:
                raise KeyError(f".npz contains {npz.files}; please provide key=")
        if key not in npz.files:
            raise KeyError(f"Key '{key}' not in {npz.files}")
        tensor = torch.from_numpy(npz[key])

    elif ext == ".mat":
        try:
            from scipy.io import loadmat
        except Exception as e:
            raise ImportError("Reading .mat requires 'scipy'. Install it or convert the file.") from e
        mdict = loadmat(path)
        # Drop meta keys that MATLAB adds
        keys = [k for k in mdict.keys() if not k.startswith("__")]
        if key is None:
            if len(keys) == 1:
                key = keys[0]
            else:
                raise KeyError(f".mat contains {keys}; please provide key=")
        if key not in mdict:
            raise KeyError(f"Key '{key}' not found in .mat. Available: {keys}")
        arr = mdict[key]
        # Ensure numeric ndarray
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Entry '{key}' in .mat is not a numpy array (type={type(arr)})")
        tensor = torch.from_numpy(arr)

    elif ext in [".txt", ".csv"]:
        # Use numpy to parse plain text; commas will be handled automatically for CSV
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        tensor = torch.from_numpy(arr)

    else:
        raise ValueError(f"Unsupported file extension '{ext}' for path: {path}")

    # Sanity check and move/cast
    if must_be_2d and tensor.dim() != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(tensor.shape)}")

    return tensor.to(device=device, dtype=dtype)
