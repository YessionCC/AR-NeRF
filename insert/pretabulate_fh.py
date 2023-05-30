from scipy import integrate
import numpy as np
import os, sys

def inte(lbd, theta_d):
  def inte_func(zeta, delta):
    return np.exp(lbd*(np.sin(zeta)*np.sin(delta) - 1))*np.sin(zeta)

  res = integrate.dblquad(inte_func, np.pi / 2 - theta_d, np.pi, 0, np.pi)
  return res[0]

def calc_inte_L(lbd): # hemisphere integral
  expTerm = 1. - np.exp(-1. * lbd)
  res = 2 * np.pi / lbd * expTerm # lx,3
  return res

def div1(lbd, theta_d):
  return inte(lbd, theta_d) / calc_inte_L(lbd)

def div2(lbd, theta_d):
  k_lbd = 0.204*lbd**3 - 0.892*lbd**2 + 2.995*lbd + 0.067 # 1,lx
  a = 1.05
  #fh_x = a / (1+k_lbd*torch.exp(ssdf)) + (1-a)/2 # px,lx
  fh_x = a / (1+np.exp(-k_lbd*theta_d)) + (1-a)/2 # px,lx
  fh_x = np.clip(fh_x, 0, 1)
  return fh_x

def pretabulate(theta_num = 1024, lbd_num = 2048, l_range_min = 0, l_range_max = 2048):
  theta_ds = np.linspace(-np.pi/2, np.pi/2, theta_num)
  lbds = np.linspace(-1, 4, lbd_num)
  lbds = 10**lbds
  res = np.ones((lbd_num, theta_num), np.float32)
  for i,lbd in enumerate(lbds[l_range_min: l_range_max]):
    print('{}: {}'.format(l_range_max, i))
    for j,theta_d in enumerate(theta_ds):
      tr = inte(lbd, theta_d)
      res[i, j] = tr
  np.save('fh_pretab_{}.npy'.format(l_range_max), res)
  print('{}: complete'.format(l_range_max))

l_range_min = int(sys.argv[1])
l_range_max = int(sys.argv[2])

pretabulate(1024, 2048, l_range_min, l_range_max)