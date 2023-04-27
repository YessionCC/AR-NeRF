import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import os, cv2
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
model_dir = './insert/generate/'

TINY_NUMBER = 1e-8

def parse_raw_sg(sg):
  SGLobes = sg[..., :3] / (torch.norm(sg[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
  SGLambdas = torch.abs(sg[..., 3:4])
  SGMus = torch.abs(sg[..., -3:])
  return SGLobes, SGLambdas, SGMus

def trans_raw_sg(sg):
  sg[..., :3] = sg[..., :3] / (torch.norm(sg[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
  sg[..., 3:4] = torch.abs(sg[..., 3:4])
  sg[..., -3:] = torch.abs(sg[..., -3:])
  return sg

viewdirs = None
def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
  global viewdirs
  if viewdirs is None:
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                            dim=-1)    # [H, W, 3]
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
  # [M, 7] ---> [..., M, 7]
  dots_sh = list(viewdirs.shape[:-2])
  M = lgtSGs.shape[0]
  lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
  # sanity
  # [..., M, 3]
  lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
  lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
  lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
  # [..., M, 3]
  rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
  rgb = torch.sum(rgb, dim=-2)  # [..., 3]
  envmap = rgb.reshape((H, W, 3))
  return envmap # H,W,3

def SG2Envmap_forDraw(lgtSGs, H, W, upper_hemi=False):
  if upper_hemi:
    phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
  else:
    phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

  viewdirs = torch.stack([
    torch.cos(theta) * torch.sin(phi), 
    torch.cos(phi), 
    torch.sin(theta) * torch.sin(phi)], dim=-1)    # [H, W, 3]
  viewdirs = viewdirs.to(lgtSGs.device)
  viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
  # [M, 7] ---> [..., M, 7]
  dots_sh = list(viewdirs.shape[:-2])
  M = lgtSGs.shape[0]
  lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
  # sanity
  # [..., M, 3]
  lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
  lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
  lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
  # [..., M, 3]
  rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
  rgb = torch.sum(rgb, dim=-2)  # [..., 3]
  envmap = rgb.reshape((H, W, 3))
  return envmap # H,W,3

def genCompareImg(im, ims, toInt8 = False):
  res = []
  for tt1, tt2 in zip(im, ims):
    tt1 = tt1.detach().cpu().numpy()
    tt2 = tt2.detach().cpu().numpy()
    tt3 = np.concatenate([tt1, tt2], 0)
    res.append(tt3)

  res = np.concatenate(res, 1)
  if toInt8:
    res = (np.clip(res, 0, 1)*255).astype(np.uint8)
  return res

def show_im(im, inGPU = True):
  plt.figure()
  if inGPU:
    plt.imshow(im.cpu().numpy())
  else:
    plt.imshow(im)
  plt.show()

def show_im_cv(im, title = 'render', waitTime = 1):
  cv2.imshow(title, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
  cv2.waitKey(waitTime)


class SGFittingNet(nn.Module):
  def __init__(self, output_sg_num = 64):
    super().__init__()
    # input 128*128*3
    self.output_sg_num = output_sg_num
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool3 = nn.MaxPool2d(2, 2)
    self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
    self.pool4 = nn.MaxPool2d(2, 2)
    self.lin = nn.Linear(256*8*8, output_sg_num*7)

  def forward(self, im):
    batch_size = im.shape[0]
    im = F.relu(self.pool1(self.conv1(im)))
    im = F.relu(self.pool2(self.conv2(im)))
    im = F.relu(self.pool3(self.conv3(im)))
    im = F.relu(self.pool4(self.conv4(im)))
    im = im.reshape(batch_size, -1)
    out = self.lin(im)
    out = out.reshape(batch_size, self.output_sg_num, 7)
    return out

class HDRIEnvData(Dataset): 
  def __init__(self, dataset_path): 
    dss = os.listdir(dataset_path)
    ims = []
    for ds in dss:
      mpath = os.path.join(dataset_path, ds)
      im = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (128, 128))
      ims.append(im)
    self.ims = np.stack(ims, 0)
    
  def __len__(self):
    return self.ims.shape[0]
  
  def __getitem__(self,index):
    return torch.Tensor(self.ims[index])

class NeRFEnvData(Dataset): 
  def __init__(self, file_path): 
    self.ims = np.load(file_path)
    
  def __len__(self):
    return self.ims.shape[0]
  
  def __getitem__(self,index):
    return torch.Tensor(self.ims[index])


class EnvTrainer:
  def __init__(self, scene_name, output_sg_num = 32, batch_size = 128, epoch = 2001):
    scene_dir = os.path.join(model_dir, scene_name)
    #dataset_path = os.path.join(scene_dir, 'envmaps')
    nerf_env_file_path = os.path.join(scene_dir, 'envmaps.npy')
    model_save_path = os.path.join(scene_dir, 'env_model')
    if not os.path.exists(model_save_path):
      os.mkdir(model_save_path)

    epo_s, model, optimizer, scheuler = self.loadckpt(model_save_path, output_sg_num)
    self.model = model
    self.ckpt_num = epo_s

    if epoch - epo_s <= 1:
      print('Env model complete training')
      self.complete_train  = True
    else:
      #data = HDRIEnvData(dataset_path)
      data = NeRFEnvData(nerf_env_file_path)
      dataloader = DataLoader(data, batch_size, True, 
        generator=torch.Generator(device='cuda'))
      self.data = data
      self.dataloader = dataloader
      self.optimizer = optimizer
      self.scheduler = scheuler
      self.model_save_path = model_save_path
      self.complete_train  = False
  
  def loadckpt(self, model_save_path, output_sg_num):
    model = SGFittingNet(output_sg_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.999))
    scheuler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)

    paths = os.listdir(model_save_path)
    model_paths = sorted([path for path in paths if 'model' in path])
    epo_s = 0
    if len(model_paths) != 0:
      model_path = model_paths[-1]
      t1 = model_path.split('.')[0]
      t2 = t1.split('_')[1]
      epo_s = int(t2)
      mpath = os.path.join(model_save_path, model_path)
      opath = os.path.join(model_save_path, 'optimizer_{}.tar'.format(t2))
      model.load_state_dict(torch.load(mpath))
      opath_data = torch.load(opath)
      optimizer.load_state_dict(opath_data['optimizer'])
      scheuler.load_state_dict(opath_data['scheuler'])
      print('Load Checkpoint: {}'.format(t2))

    return epo_s, model, optimizer, scheuler

  def train(self, epoch = 10000, visual = False):
    if self.complete_train:
      return
    print('Train env model ...')
    self.model.train()
    for epo in range(self.ckpt_num, epoch):

      for im in self.dataloader:
        #cv2.imwrite("t.png", (im[0].detach().cpu().numpy()*255).astype(np.uint8))
        self.optimizer.zero_grad()
        out = self.model(im.permute(0,3,1,2))

        ims = []
        for i in range(im.shape[0]):
          res = SG2Envmap(out[i], 128, 128)
          ims.append(res)
        ims = torch.stack(ims, 0)

        loss = F.mse_loss(ims, im)
        loss.backward()
        self.optimizer.step()
      
      self.scheduler.step()
      print('Epoch: {}, loss: {}'.format(epo, loss.item()))
      
      if visual:
        tcmp = genCompareImg(im[:4], ims[:4], True)
        show_im_cv(tcmp, waitTime=1)
      if epo % 500 == 0 and epo != 0:
        tcmp = genCompareImg(im[:4], ims[:4], True)
        impath = os.path.join(self.model_save_path, 'res_im_{:04d}.png'.format(epo))
        mpath = os.path.join(self.model_save_path, 'model_{:06d}.tar'.format(epo))
        opath = os.path.join(self.model_save_path, 'optimizer_{:06d}.tar'.format(epo))
        cv2.imwrite(impath, tcmp)
        torch.save(self.model.state_dict(), mpath)
        torch.save({
          'optimizer': self.optimizer.state_dict(),
          'scheuler': self.scheduler.state_dict()
        }, opath)

  def eval_mode(self):
    if not self.complete_train:
      self.data = None
      self.dataloader = None
      self.optimizer = None
      self.scheduler = None
    self.model.eval()

  # im: H,W,3
  @torch.no_grad()
  def eval(self, im):
    #show_im(im)
    im = im.permute(2,0,1)
    res = self.model(im[None, ...])[0]
    #show_im(SG2Envmap(res, 128, 128))
    return res


class EnvOptim:
  def __init__(self, numLgtSGs = 32, N_iter = 25):
    lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
    lgtSGs.data[..., 3:4] *= 100.
    lgtSGs.requires_grad = True
    self.lgtSGs = lgtSGs
    self.optimizer = torch.optim.Adam([self.lgtSGs,], lr=1e-1)
    self.N_iter = N_iter
  
  def eval(self, im, visual = False):
    H, W = im.shape[:2]
    for step in range(self.N_iter):
      self.optimizer.zero_grad()
      env_map = SG2Envmap(self.lgtSGs, H, W)
      loss = F.mse_loss(env_map, im)
      loss.backward()
      self.optimizer.step()

      if visual:
        tcmp = genCompareImg(env_map[None,...], im[None,...], True)
        show_im_cv(tcmp, waitTime=1, title='cmp')
      
    return self.lgtSGs.detach()
