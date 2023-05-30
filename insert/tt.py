import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2

import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def optim():
  params = torch.rand(3)
  params = nn.parameter.Parameter(params)
  optimizer = torch.optim.Adam(
    [{'params': params}], 
    lr=1e-2, betas=(0.9, 0.999),
  )

  for _ in range(1000000):
    x1 = torch.rand(4096*16) * 0.4 # 0~0.4
    x2 = torch.rand(4096*4) * 100 + 10 # 10~110

    y_dst1 = torch.pow(x1 / (1+x1), 1/2.2)
    y_src1 = torch.log(x1+params[0])*params[1]+params[2]

    y_dst2 = torch.log(1+x2)
    y_src2 = torch.log(x2+params[0])*params[1]+params[2]

    y0 = torch.log(params[0])*params[1]+params[2]

    optimizer.zero_grad()
    loss = 50*F.mse_loss(y_src1, y_dst1) + \
      F.mse_loss(y_src2, y_dst2) + \
      100*y0**2
    loss.backward()
    optimizer.step()
    print('params: {}, loss: {}'.format(params, loss.item()))

#optim()

ps = [0.2935, 0.7607, 0.9325]

x = np.arange(0, 1, 0.001)
y1 = np.power(x/(1+x), 1/2.2)
#y2 = np.log(x+0.0332)*0.1925+0.6555
y2 = np.log(x+ps[0])*ps[1]+ps[2]
print(np.log(ps[0])*ps[1]+ps[2])
y3 = np.tanh(x)
y4 = np.log(1+x)

plt.figure()
plt.plot(x,y1,label='hdr')
plt.plot(x,y2,label='log')
plt.plot(x,y3,label='tanh')
plt.plot(x,y4,label='loge')
plt.legend()
plt.show()