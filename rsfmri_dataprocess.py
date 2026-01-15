import numpy as np
from sklearn.linear_model import LinearRegression

def _no_modify(data, *args, **kwargs):
    return data

def _csf_mean(data, csf_voxel=[(50, 55, 47),(39, 55, 47)]):
    return np.mean([data[x,y,z] for x,y,z in csf_voxel], axis = 0)

def _regress_csf(data,csf_method = _csf_mean):
    assert data.ndim == 4, "Expected 4D fMRI data"
    shape = data.shape
    csf_mean = csf_method(data)
    flat_data = data.reshape(-1, shape[3])  # shape: (91*109*91, 150)
    
    # Fit linear regression and remove CSF component
    reg = LinearRegression()
    reg.fit(csf_mean.reshape(-1, 1), flat_data.T)
    predicted = reg.predict(csf_mean.reshape(-1, 1)).T
    residuals = flat_data - predicted
    
    # Reshape back to original shape
    denoised_data = residuals.reshape(shape)
    return denoised_data

def _rem_scale(data, scale = 10_000, csf_method = _csf_mean):
    csf_scale = np.mean(csf_method(data))
    return data/csf_scale*scale