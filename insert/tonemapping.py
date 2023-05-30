import torch
import numpy as np
import cv2


def tonemapping_simple_np_log(im, **kwargs):
  return np.log(1. + 5000. * im) / np.log(1. + 5000.)

def tonemapping_simple_torch_log(im, **kwargs):
  return torch.log(1. + 5000. * im) / np.log(1. + 5000.)

def tonemapping_simple_np_gamma(im, **kwargs):
  return np.power(im / (1+im), 1.0/2.2)

def tonemapping_simple_torch_gamma(im, **kwargs):
  return torch.pow(im / (1+im), 1.0/2.2)

def tonemapping_simple_np_linear(im, **kwargs):
  return np.power(np.clip(im, 0, 1), 1.0/2.2)

def tonemapping_simple_torch_linear(im, **kwargs):
  return torch.pow(im.clip(0,1), 1.0/2.2)

def tonemapping_complex_np_Reinhard(im, **kwargs):
  tonemapReinhard = cv2.createTonemapReinhard(2.2, 1,0.5,0)
  return tonemapReinhard.process(im)

def tonemapping_complex_torch_Reinhard(im, **kwargs):
  tonemapReinhard = cv2.createTonemapReinhard(2.2, 1,0.5,0)
  return torch.Tensor(tonemapReinhard.process(im.cpu().numpy()))

tonemapping_simple_np = tonemapping_simple_np_gamma
tonemapping_simple_torch = tonemapping_simple_torch_gamma
#tonemapping_complex = tonemapping_complex_Reinhard