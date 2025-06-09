"""
torchrtm.utils.torch_utils
--------------------------

General-purpose torch utilities.
"""

import torch
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

def Torchlut(simulator = '', table_size=500000, std=0, units=10000, wavelength=[],d = 6):
    # Initialize Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=d)
    samples = sampler.random(n=table_size)

    ref_list = []
    para_list = []

    # Determine the total number of iterations
    num_iterations = max(1, int(table_size / units))

    # Loop to process samples in batches with a progress bar
    for ii_index in tqdm(range(num_iterations), desc="Processing Batches"):
        test_num = units

        # Generate a subset of samples and convert to PyTorch tensor
        many_paras = torch.tensor(samples[units * ii_index:units * (ii_index + 1), :]).to(torch.float32)
        if std > 0:
            many_paras += torch.randn_like(many_paras) * std

        # Example normalization function placeholder; replace with actual normalization if needed
        real_paras = normlization_torch(many_paras, fitting=False).to(device)

        # Pass through prospect5b model
        spec_data = simulator(real_paras[:, 1:], real_paras[:, 0])[0] ## here should has a bug

        para_list.append(real_paras.cpu().numpy())
        ref_list.append(spec_data.cpu().numpy())
        real_paras = spec_data = []

    # Combine lists into arrays
    para_list = np.vstack(para_list)
    ref_list = np.vstack(ref_list)
    if len(wavelength)>0:
      return ref_list[:,wavelength-400], para_list[:,wavelength-400]

    else:
      return ref_list, para_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normlization_torch(x,x_max = np.array([5,197.6048780487804, 56.7, 5,0.1049756097560976, 0.0314634146341464]),
                       x_min = np.array([1,0, 0,0,0, 0]), fitting = True):
  x_max = torch.tensor(x_max).to(torch.float32)
  x_min = torch.tensor(x_min).to(torch.float32)


  try:
    x2 = x.clone()
  except:
    x2 = x.copy()
  if (len(x2.shape)) == 2:

    if fitting == True:
      for _ in range(x2.shape[-1]):
        x2[:,_] = (x[:,_] - x_min[_])/(x_max[_] - x_min[_])
    else:
      for _ in range(x2.shape[-1]):
        x2[:,_] = x[:,_]*(x_max[_] - x_min[_]) + x_min[_]
  else:
    if fitting == True:
      for _ in range(6):
        x2[:,_] = (x[:,_] - x_min[_])/(x_max[_] - x_min[_])
    else:
      for _ in range(6):
        x2[:,_] = x[:,_]*(x_max[_] - x_min[_]) + x_min[_]
  return(x2)


