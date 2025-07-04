"""
torchrtm.canopy.suits
---------------------

Implements the SUITS directional scattering approximation used in SAIL.
Includes the volume scattering functions used in SUITS approximation for canopy radiative transfer.
"""

import torch
import torch._dynamo


def SUITS(na, litab, lidFun, tts, tto, cts, cto, psi, ctscto,ks=0, ko=0, bf=0, sob=0, sof=0,
          len_batch=0, len_na=13):
    """
    Computes scattering parameters for SAIL using the SUITS approximation.

    Args:
        na (torch.Tensor): Dummy tensor used for dimension tracking (e.g., angular bins).
        litab (torch.Tensor): Leaf inclination bin centers [deg].
        lidFun (torch.Tensor): LIDF weights, shape (B, 13) or (13,).
        tts (torch.Tensor): Solar zenith angle [deg], shape (B,) or scalar.
        tto (torch.Tensor): Observer zenith angle [deg], shape (B,) or scalar.
        cts (torch.Tensor): cos(tts), shape (B,) or scalar.
        cto (torch.Tensor): cos(tto), shape (B,) or scalar.
        psi (torch.Tensor): Relative azimuth angle [deg], shape (B,) or scalar.
        ctscto (torch.Tensor): cos(tts) * cos(tto), shape (B,) or scalar.
        use_batch (bool): Whether to apply batched processing.
        ks, ko, bf, sob, sof (float): Initial scattering terms.
        len_batch (int): Batch size (only required if use_batch=True).
        len_na (int): Number of angular bins (default 13).

    Returns:
        list: [ks, ko, sob, sof, sdb, sdf, dob, dof, ddb, ddf]
    """
    pi = torch.tensor(torch.pi, device=cts.device)
    rd = pi / 180.0
    ctl = torch.cos(rd * litab)

    chi_s, chi_o, frho, ftau = volscatt(tts, tto, psi, litab, len_batch, len_na)


    ksli = chi_s / cts.unsqueeze(1)
    koli = chi_o / cto.unsqueeze(1)
    sobli = frho * pi / ctscto.unsqueeze(1)
    sofli = ftau * pi / ctscto.unsqueeze(1)
    bfli = ctl**2

    ks = (ksli * lidFun).sum(dim=1)
    ko = (koli * lidFun).sum(dim=1)
    bf = (bfli.unsqueeze(0) * lidFun).sum(dim=1)
    sob = (sobli * lidFun).sum(dim=1)
    sof = (sofli * lidFun).sum(dim=1)

    sdb = 0.5 * (ks + bf)
    sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1 + bf)
    ddf = 0.5 * (1 - bf)

    return [ks, ko, sob, sof, sdb, sdf, dob, dof, ddb, ddf]


# Helper function: Volume Scattering base on Scattering by Arbitrarily Inclined Leaves


def volscatt(tts, tto, psi, ttl, len_batch=0,len_na = 13):
    """
    Computes the scattering coefficients (chi_s, chi_o, frho, and ftau) for the SUITS model.

    Parameters:
    - tts: Incident angle (sensor to surface).
    - tto: Outgoing angle (surface to sensor).
    - psi: Relative angle between incident and reflected rays.
    - ttl: Lookup table for angles.
    - len_batch: The batch size for batch processing (default is 0 for no batch).

    Returns:
    - chi_s: Scattering coefficient for the sensor.
    - chi_o: Scattering coefficient for the outgoing sensor.
    - frho: Reflection coefficient.
    - ftau: Transmission coefficient.
    """
    device=ttl.device
    # Initialize constants
    pi = torch.tensor(torch.pi, device=device)
    rd = pi / 180
    ctl = torch.cos(rd * ttl)

    # Calculate trigonometric terms for incident and outgoing angles
    costs = torch.cos(rd * tts)
    costo = torch.cos(rd * tto)
    sints = torch.sin(rd * tts)
    sinto = torch.sin(rd * tto)
    cospsi = torch.cos(rd * psi)

    psir = rd * psi  # Converted psi to radians

    # Calculate additional terms
    costl = torch.cos(rd * ttl)
    sintl = torch.sin(rd * ttl)



    # Expand tensors for batch processing
    cs = (costs * costl.expand(len_batch, len_na).T).T
    co = (costo * costl.expand(len_batch, len_na).T).T
    ss = (sints * sintl.expand(len_batch, len_na).T).T
    so = (sinto * sintl.expand(len_batch, len_na).T).T
    tto = tto.expand(len_na, len_batch).T

    # Initialize result tensors for batch processing
    bt1 = torch.zeros(len_batch, len_na).to(device)
    bt2 = torch.zeros(len_batch, len_na).to(device)
    bt3 = torch.zeros(len_batch, len_na).to(device)
    doo = torch.zeros(len_batch, len_na).to(device)
    bts = torch.zeros(len_batch, len_na).to(device)
    bto = torch.zeros(len_batch, len_na).to(device)
    ds = torch.zeros(len_batch, len_na).to(device)
    cosbts = torch.zeros(len_batch, len_na).to(device) + 5.0
    cosbto = torch.zeros(len_batch, len_na).to(device) + 5.0
    band_condi = torch.zeros(len_batch, len_na).to(device)
    psir_matrix = psir.expand(len_na, len_batch).T
    t2 = torch.zeros(len_batch, len_na).to(device)

    # Calculate cosine values based on conditions
    condi_cosbts = torch.abs(so) > 1e-6
    cosbts[condi_cosbts] = -cs[condi_cosbts] / ss[condi_cosbts]

    # Compute more scattering coefficients
    condi_cosbto = torch.abs(so) > 1e-6
    cosbto[condi_cosbto] = -co[condi_cosbto] / so[condi_cosbto]

    # Handle edge cases for values of cosbts
    condi_cosbts_1 = torch.abs(cosbts) < 1
    bts[condi_cosbts_1] = torch.acos(cosbts[condi_cosbts_1])
    ds[condi_cosbts_1] = ss[condi_cosbts_1]

    bts[~condi_cosbts_1] = pi
    ds[~condi_cosbts_1] = cs[~condi_cosbts_1]

    chi_s = 2 / pi * ((bts - pi * 0.5) * cs + torch.sin(bts) * ss)

    # Compute outgoing scattering coefficients
    condi_cosbto_1 = torch.abs(cosbto) < 1
    if len_batch == 0:
        band_condi = torch.zeros(len_na.to(device))
    else:
        band_condi = torch.zeros(len_batch, len_na).to(device)

    bto[condi_cosbto_1] = torch.acos(cosbto[condi_cosbto_1])
    doo[condi_cosbto_1] = so[condi_cosbto_1]
    band_condi[condi_cosbto_1] = 1

    band_condi[(band_condi == 0) & (tto < 90)] = 2
    condi_tto_90 = band_condi == 2
    bto[condi_tto_90] = pi
    doo[condi_tto_90] = co[condi_tto_90]
    band_condi[condi_cosbto_1] = 1

    bto[band_condi == 0] = 0
    doo[band_condi == 0] = -co[band_condi == 0]

    chi_o = 2 / pi * ((bto - pi * 0.5) * co + torch.sin(bto) * so)

    # Compute transition coefficients and reflection/transmission
    btran1 = torch.abs(bts - bto)
    btran2 = pi - torch.abs(bts + bto - pi)
    if len_batch == 0:
        band_condi = torch.zeros(len_na.to(device)) + 1
        condi_psir_btran1 = psir <= btran1
        bt1[condi_psir_btran1] = psir
    else:
        band_condi = torch.zeros(len_batch, len_na).to(device) + 1
        condi_psir_btran1 = psir_matrix <= btran1
        bt1[condi_psir_btran1] = psir_matrix[condi_psir_btran1]
        psir = psir_matrix

    bt2[condi_psir_btran1] = btran1[condi_psir_btran1]
    bt3[condi_psir_btran1] = btran2[condi_psir_btran1]
    band_condi[condi_psir_btran1] = 0
    bt1[band_condi != 0] = btran1[band_condi != 0]

    # Compute final coefficients for reflection and transmission
    btran1[(band_condi != 0) & (psir <= btran2)] = 2
    if len_batch == 0:
        bt2[band_condi == 2] = psir[band_condi == 2]
        bt3[band_condi == 2] = btran2[band_condi == 2]

    frho = 1.0 * btran1
    ftau = 1.0 * btran2

    # Return scattering coefficients
    return chi_s, chi_o, frho, ftau
