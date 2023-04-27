from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .colmap_exr import ColmapExrDataset
from .myblender import MyBlender
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'colmap_exr': ColmapExrDataset,
                'myblender': MyBlender,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset}