import numpy as np
from scipy.fft import fft, fftfreq


def _voxel_mean(data):
    return np.mean(data,axis = -1)

def _voxel_sd(data,ddof=1):
    return np.std(data, axis = -1,ddof=ddof)

# Total time=Number of timepoints×Repetition Time (TR)
def _voxel_alff(data,mask = None,tr = 3.0,low=0.01, high=0.08):
    N = data.shape[-1]
    fft_vals = fft(data, axis=-1)
    freqs = fftfreq(N, d=tr)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    ampl = np.abs(fft_vals[..., pos_mask])/N
    band_mask = (freqs >= low) & (freqs <= high)
    alff = np.mean(ampl[...,band_mask],axis = -1)
    if len(alff.shape) == 1:
        alff = alff/np.nanmean(alff)
    elif len(alff.shape) == 3:
        alff = alff / np.nanmean(alff[mask])
    else:
        raise Exception("Error in ALFF shape")
    return alff

def _voxel_falff(data, tr=3.0,low=0.01, high=0.08):
    N = data.shape[-1]
    fft_vals = fft(data, axis=-1)
    freqs = fftfreq(N, d=tr)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    ampl = np.abs(fft_vals[..., pos_mask])/N
    band_mask = (freqs >= low) & (freqs <= high)
    frac = np.nan_to_num(np.sum(ampl[...,band_mask],axis = -1))
    frac[frac == 0] = 1
    alff = frac/np.sum(ampl,axis = -1)
    return alff

def _vox_fft(data, tr=3.0, low=0.01, high=0.08):
    N = data.shape[-1]
    fft_vals = fft(data, axis=-1)
    freqs = fftfreq(N, d=tr)
    pos_mask = (freqs >= low) & (freqs <= high)
    freqs = freqs[pos_mask]
    power = (np.abs(fft_vals[..., pos_mask])/N)**2
    return power,freqs

def _voxel_spec_cent(data, tr=3.0, low=0.01, high=0.08):
    power,freqs = _vox_fft(data, tr=tr, low=low,high=high)
    with np.errstate(divide='ignore', invalid='ignore'):
        SC = np.sum(power*freqs,axis = -1)/np.sum(power,axis=-1)
    return SC

def _voxel_spec_var(data, tr=3.0, low=0.01, high=0.08):
    power,freqs = _vox_fft(data, tr=tr, low=low,high=high)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = power/np.sum(power,axis=-1,keepdims=True)
        sc = _voxel_spec_cent(data)[..., np.newaxis]
        f_arr = freqs[np.newaxis, np.newaxis, np.newaxis, :] - sc
        SV = np.sum(p*f_arr**2,axis=-1)
    return np.squeeze(SV)

def _voxel_spec_skewness(data, tr=3.0, low=0.01, high=0.08):
    power,freqs = _vox_fft(data, tr=tr, low=low,high=high)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = power/np.sum(power,axis=-1,keepdims=True)
        sc = _voxel_spec_cent(data)[..., np.newaxis]
        f_arr = freqs[np.newaxis, np.newaxis, np.newaxis, :] - sc
        m3 = np.sum(p*f_arr**3,axis=-1)
        m2 = np.sum(p*f_arr**2,axis=-1)
        SS = m3/(m2**(3/2))
    return np.squeeze(SS)