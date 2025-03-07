import torch


def get_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    elif dtype_name == "float64":
        return torch.float64
    elif dtype_name == "float16":
        return torch.float16
    elif dtype_name == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Dtype {dtype_name} not found")
