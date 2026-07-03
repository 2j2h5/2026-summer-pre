import torch


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> None:
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        try:
            props = torch.cuda.get_device_properties(0)
            print("VRAM GB:", round(props.total_memory / 1024**3, 2))
        except Exception:
            pass
