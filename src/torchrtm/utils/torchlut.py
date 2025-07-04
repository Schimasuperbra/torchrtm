"""
torchrtm.utils.torch_utils
--------------------------

General-purpose torch utilities.
"""

import torch
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Torchlut(param_type='prosail', table_size=500000, std=0, batch=10000, wavelength=None,d = None,sensor_name = 'LANDSAT4-TM',sail_prospect = 'prospectd'):
    # Initialize Latin Hypercube Sampler

    if d is None:
      param_dims = {
          'atom': 18,
          'prosail': 15,
          'prospectd': 7,  
          'prospect5': 6   
      }
      d = param_dims.get(param_type, 6)



    if param_type == 'atom':
        simulator = prosail
        from torchrtm.atmosphere.smac import smac
        from torchrtm.data_loader import load_smac_sensor
        from torchrtm.atmosphere.smac import toc_to_toa

    elif param_type == 'prosail':
        simulator = prosail
    elif param_type == 'prospect5b':
        simulator = prospect5b
    elif param_type == 'prospectd':
        simulator = prospectd

    sampler = qmc.LatinHypercube(d=d)
    samples = sampler.random(n=table_size)
    ref_list = []
    para_list = []
    # Determine the total number of iterations
    num_iterations = max(1, int(table_size / batch))
    # Loop to process samples in batches with a progress bar
    for ii_index in tqdm(range(num_iterations), desc="Processing Batches"):
        test_num = batch
        # Generate a subset of samples and convert to PyTorch tensor
        many_paras = torch.tensor(samples[batch * ii_index:batch * (ii_index + 1), :]).to(torch.float32)
        if std > 0:
            many_paras += torch.randn_like(many_paras) * std
        # Example normalization function placeholder; replace with actual normalization if needed
        if param_type == 'atom':
          many_paras[:,:15] = normalize_parameters(many_paras[:,:15],param_type='prosail', fitting=False).to(device)
          many_paras[:,15:] = normalize_parameters(many_paras[:,15:],param_type=param_type, fitting=False).to(device)

        #real_paras = normlization_torch(many_paras, param_type=param_type, fitting=False)

        # Pass through prospect5b model
        if param_type in ['prospectd', 'prospect5b']:
            # For PROSPECT models: typically (leaf_params, LAI) or similar structure
            spec_data = simulator(many_paras[:, 1:], many_paras[:, 0],device = device)[0]

        else:
            lai = many_paras[:,0]
            LIDFa = many_paras[:,1]
            LIDFb = many_paras[:,2]
            q = many_paras[:,3]
            tts = many_paras[:,4]
            tto = many_paras[:,5]
            psi = many_paras[:,6]
            psoil = many_paras[:,7]
            N = many_paras[:,8]
            Cab = many_paras[:,9]
            Car = many_paras[:,10]
            Cbrown = many_paras[:,11]
            Cw = many_paras[:,12]
            Cm = many_paras[:,13]
            Canth = many_paras[:,14]
            alpha = torch.tensor(40).to(device).expand([test_num])
            tran_alpha = alpha.clone()
            if sail_prospect == 'prospectd':
                traits = torch.stack([Cab,Car,Cbrown,Cw,Cm,Canth],axis = 1)
                spec_data = simulator(traits,N,LIDFa,LIDFb,lai,q,tts,tto,psi,alpha,psoil,batch_size=1,use_prospectd=True,lidtype=2)
            else:
                traits = torch.stack([Cab,Car,Cbrown,Cw,Cm],axis = 1)
                spec_data = simulator(traits,N,LIDFa,LIDFb,lai,q,tts,tto,psi,alpha,psoil,batch_size=1,use_prospectd=True,lidtype=2)
        if param_type == 'atom':
            aot550 = atom_papra[:,-3]
            uo3 = atom_papra[:,-2]
            uh2o = atom_papra[:,-1]
            coefs,sm_wl = load_smac_sensor(sensor_name.split('.')[0])
            Ta_s, Ta_o, T_g, ra_dd, ra_so, ta_ss, ta_sd, ta_oo, ta_do = smac(tts,tto,psi,coefs)
            # return to the R_TOA
            R_TOC, R_TOA = toc_to_toa(spec_data.permute(0, 2, 1), sm_wl-400, ta_ss, ta_sd, ta_oo, ta_do, ra_so, ra_dd, T_g, return_toc=True)
            #toa_results.append([sm_wl,R_TOA])
            spec_data = R_TOA
        else:
          spec_data=spec_data[3]
        #param_type in ['prospectd', 'prospect5b']:     
        para_list.append(many_paras.cpu().numpy())
        ref_list.append(spec_data.cpu().numpy())
        many_paras = spec_data = []
    # Combine lists into arrays
    para_list = np.vstack(para_list)
    ref_list = np.vstack(ref_list)
    if len(wavelength)>0:
      return ref_list[:,wavelength-400], para_list[:,wavelength-400]
    else:
      return ref_list, para_list
