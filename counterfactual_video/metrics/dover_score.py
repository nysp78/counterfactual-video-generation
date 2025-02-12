'''Adapted from  https://github.com/VQAssessment/DOVER/tree/master'''

import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
import torch.nn as nn
from .dover.models import DOVER
from torchvision.transforms import ToPILImage
from .utils.dover_utils import DoverPreprocessor, fuse_results

logger = logging.getLogger(__name__)

class DoverScore(nn.Module):
    pretrained_config_file = 'metrics/dover/config.yaml'
    pretrained_checkpoint = 'metrics/checkpoints/DOVER/DOVER.pth'

    def __init__(
        self,
        device: torch.device = "cuda",
        pretrained_config_file: str = None,
        pretrained_checkpoint: str = None,
    ):
        super().__init__()
        pretrained_config_file = pretrained_config_file or self.pretrained_config_file
        pretrained_checkpoint = pretrained_checkpoint or self.pretrained_checkpoint

        logger.debug(f"Loding model {pretrained_checkpoint}")
        config = OmegaConf.to_container(OmegaConf.load(pretrained_config_file))
        self.preprocessor = DoverPreprocessor(config["data"])
        self.device = device
        self.model = DOVER(**config['model'])
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(pretrained_checkpoint, map_location=self.device))
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def range(self):
        return 0, 1

    def preprocess(self, video):
        views = self.preprocessor(video)
        for k, v in views.items():
            views[k] = v.to(self.device)
        return views
 
    @torch.no_grad()
    def evaluate(self, video) -> float:
       # print(views.shape)
        #video = (video * 255).byte()  # Convert to uint8 (integer values)
        to_pil = ToPILImage()  # Define transformation
        pil_video = [to_pil(frame) for frame in video]
        views = self.preprocess(pil_video)
        results = [r.mean().item() for r in self.model(views)]
        # score
        scores = fuse_results(results)
        return scores