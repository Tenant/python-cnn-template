from .kitti import KITTIDataset
from .sceneflow import SceneFlowDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset
}

