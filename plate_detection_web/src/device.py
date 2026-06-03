def resolve_inference_device(device):
    value = str(device or "auto").strip().lower()

    if value in {"cpu", "mps"}:
        return value

    if value.isdigit():
        return int(value)

    if value.startswith("cuda:"):
        _, index = value.split(":", 1)
        return int(index) if index.isdigit() else 0

    if value in {"auto", "gpu", "cuda"}:
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    return device


def device_label(device):
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)
