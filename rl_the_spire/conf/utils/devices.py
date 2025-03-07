import torch


def get_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "cuda":
        return torch.device("cuda")
    elif device_name == "mps":
        return torch.device("mps")
    else:
        raise ValueError(f"Device {device_name} not found")
