from pathlib import Path
import torch

from MTCNN.models.pnet import PNet
from MTCNN.models.rnet import RNet
from MTCNN.models.onet import ONet

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def load_pretrained_models(device="cpu", weights_dir=None):
    directory = Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR
    models = []

    for name, cls in (("pnet", PNet), ("rnet", RNet), ("onet", ONet)):
        path = directory / f"{name}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy weights: {path}")

        model = cls()
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        models.append(model.to(device).eval())

    return tuple(models)
