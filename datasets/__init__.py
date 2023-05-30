from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .colmap_exr import ColmapExrDataset
from .colmap_real_exr import ColmapRealExrDataset
from .myblender import MyBlender
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'colmap_exr': ColmapExrDataset,
                'colmap_real_exr': ColmapRealExrDataset,
                'myblender': MyBlender,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset}