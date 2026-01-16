import gc
import numpy as np
from tqdm import tqdm

import nibabel as nib
from nilearn.glm import threshold_stats_img
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

def _combine_mtd(data,mtd_lis,*args, **kwargs):
    val_lis = []
    for mtd in mtd_lis:
        val_lis.append(mtd(data,*args, **kwargs))
    return np.stack(val_lis,axis = -1)

def _voxel_data(method, c_datalis, d_datalis, remove_csf, scale, mask,
                shape = (91,109,91), full = False, dtype = np.float32,
                *args, **kwargs):
    mask_voxel = np.sum(mask)
    nc, nd = len(c_datalis), len(d_datalis)
    c_arr = np.empty((nc,mask_voxel),dtype= dtype)
    for i, c_data in enumerate(tqdm(c_datalis)):
        c_niobj = scale(c_data.get_fdata(dtype=dtype))
        c_niobj = remove_csf(c_niobj)[mask]
        c_arr[i] = method(c_niobj, *args, **kwargs)
        del c_niobj, c_data
        gc.collect()
    d_arr = np.empty((nd,mask_voxel),dtype= dtype)
    for i, d_data in enumerate(tqdm(d_datalis)):
        d_niobj = scale(d_data.get_fdata(dtype=dtype))
        d_niobj = remove_csf(d_niobj)[mask]
        d_arr[i] = method(d_niobj, *args, **kwargs)
        del d_niobj, d_data
        gc.collect()
    if not full:
        return c_arr,d_arr
    V = np.prod(shape)
    shape_c, shape_d = (nc,V), (nd,V)
    flat_mask = mask.ravel()
    c_arr_full, d_arr_full = np.full(shape_c, np.nan), np.full(shape_d, np.nan)
    c_arr_full[:,flat_mask], d_arr_full[:,flat_mask] = c_arr,d_arr
    full_shape_c, full_shape_d = (nc,)+shape, (nd,)+shape
    return c_arr_full.reshape(full_shape_c),d_arr_full.reshape(full_shape_d)


def _shuffle_columns(arr):
    if arr.ndim not in (2, 3):
        raise ValueError("Input must be 2D or 3D array.")
    rows, cols = arr.shape[0], arr.shape[1]
    idx = np.argsort(np.random.rand(rows, cols), axis=0)  # random permutations    
    if arr.ndim == 2:
        return arr[idx, np.arange(cols)]
    else:  # arr.ndim == 3
        depth = arr.shape[2]
        shuffled = np.empty_like(arr)
        for d in range(depth):
            idx = np.argsort(np.random.rand(rows, cols), axis=0)
            shuffled[..., d] = arr[idx, np.arange(cols),d]
        return shuffled

def _welch_ttest(x,y): #for flat data, e.g.
    n1, n2 = np.sum(~np.isnan(x),axis=0), np.sum(~np.isnan(y),axis=0)
    mean1, mean2 = np.nanmean(x, axis = 0), np.nanmean(y,axis = 0)
    var1, var2 = np.nanvar(x, ddof=1,axis =0), np.nanvar(y, ddof=1, axis=0)
    denom = np.sqrt(var1/n1 + var2/n2)
    denom[denom==0] = 1
    t = (mean1 - mean2) / denom
    return t

def _perm_test(x,y, nsim = 100, test_stat = _welch_ttest):
    # stacked data
    # ensure reshaped data to n_sample, Voxels, features
    nx = len(x)
    act_t = test_stat(x,y)
    arr = np.vstack((x,y))
    V = x.shape[1]
    p_val = np.zeros(V)
    for _ in tqdm(range(nsim)):
        shuf_arr = _shuffle_columns(arr)
        t_stat = test_stat(shuf_arr[:nx],shuf_arr[nx:])
        p_val += (np.abs(t_stat) >= np.abs(act_t))
        del shuf_arr, t_stat
        gc.collect()
    p_val = (1 + p_val) / (1 + nsim)
    gc.collect()
    return act_t,p_val

def _voxel_perm_test(method, c_datalis, d_datalis, remove_csf, scale, mask,
                    shape = (91,109,91), dtype = np.float32, nsim = 1000, 
                    test_stat = _welch_ttest,
                    *args, **kwargs):
    # print(locals())
    c_arr, d_arr = _voxel_data(method, c_datalis, d_datalis, remove_csf, scale, mask,
                               shape, dtype=dtype,
                               *args, **kwargs)
    V = np.prod(shape)
    t_stat,p_val = _perm_test(c_arr, d_arr,nsim,test_stat)
    del c_arr,d_arr
    t_stat_full, p_val_full = np.full(V, np.nan), np.full(V, np.nan)
    flat_mask = mask.ravel()
    t_stat_full[flat_mask], p_val_full[flat_mask] = t_stat, p_val
    t_stat_full, p_val_full = t_stat_full.reshape(shape), p_val_full.reshape(shape)
    z_score_full = norm.isf(p_val_full) #one-sided z-score 
    return t_stat_full, p_val_full, z_score_full

def _zmap_thres(z_map, alpha = 0.05, cluster=10, dtype = np.float32,
                two_sided=False, height_control = 'fdr', atlas = None):
    reference_img = atlas.maps
    t_img = nib.Nifti1Image(z_map.astype(np.float64), reference_img.affine)
    thresholded_map2, threshold2 = threshold_stats_img(t_img, alpha=alpha, 
                                                       height_control=height_control,
                                                       cluster_threshold=cluster,
                                                       two_sided=two_sided)
    print(f"The FDR=.05 threshold is {threshold2:.3g}")
    z_stat_corrected = thresholded_map2.get_fdata(dtype = dtype)
    return z_stat_corrected

def _region_correction(p_val, atlas, control = 'fdr_bh', alpha = 0.05, re_dic = False,
                        standard_mask = None, cluster = None):
    VALID_CONTROLS = {'bonferroni', 'holm', 'fdr_bh', 'fdr_by', 'sidak', 'hommel'}
    if control not in VALID_CONTROLS:
        raise ValueError(f"Invalid control method '{control}'. Must be one of: {', '.join(VALID_CONTROLS)}")
    atlas_data = atlas.maps.dataobj
    p_corrected = np.empty(p_val.shape)
    p_corrected[:] = np.nan
    for i in range(len(atlas.labels[1:])):
        mask = atlas_data == i+1
        mask*= standard_mask
        p_val_reg = p_val[mask]
        corrected_pvals = multipletests(p_val_reg, alpha=alpha, method=control)[1]
        p_corrected[mask] = corrected_pvals
    if cluster is None:
        z_corrected = norm.isf(p_corrected)
        return p_corrected, z_corrected
    z_score = norm.isf(p_corrected)
    z_cor_clust = _zmap_thres(z_score,alpha=alpha,cluster=cluster,height_control='fpr')
    p_corr_clust = norm.sf(z_cor_clust)
    return p_corr_clust, z_cor_clust

def _pval_percent(p_val,atlas = None, standard_mask = None):
    if atlas is not None:
        atlas_data = atlas.maps.dataobj
        reg_name = atlas.labels[1:]
        region_data = {}  
        for i in range(len(reg_name)):
            mask = atlas_data == i+1
            mask*= standard_mask
            p_val_reg = p_val[mask]
            region_data[reg_name[i]] = (np.nansum(p_val_reg<0.05)/np.sum(mask)).item()
        return region_data
    else:
        return (np.nansum(p_val<0.05)/np.sum(standard_mask)).item() #for total

def _region_perm_test(mtd_lis, atlas, c_datalis, d_datalis, remove_csf, scale, 
                      mask, dtype=np.float32, nsim = 1_000, test_stat = _welch_ttest,
                      *args, **kwargs):
    atlas_data = atlas.maps.dataobj
    region_labels = atlas.labels[1:]
    n_reg = len(region_labels)
    c_arr = np.empty((len(mtd_lis),len(c_datalis),n_reg),dtype= dtype)
    print("Control data")
    for i, c_data in enumerate(tqdm(c_datalis)):
        c_niobj = scale(c_data.get_fdata(dtype=dtype))
        c_niobj = remove_csf(c_niobj)
        N = c_niobj.shape[-1]
        c_region_arr = np.zeros((n_reg,N))
        for region_index, region_name in enumerate(region_labels):
            reg_mask = atlas_data == region_index+1
            reg_mask*= mask
            c_region_arr[region_index] = np.nanmean(c_niobj[reg_mask],axis = 0) 
        del c_niobj
        for j, method in enumerate(mtd_lis):
           c_arr[j,i] = method(c_region_arr, *args, **kwargs)
        del c_region_arr
        gc.collect()
    print("Alternative data")
    d_arr = np.empty((len(mtd_lis),len(d_datalis),n_reg),dtype= dtype)
    for i, d_data in enumerate(tqdm(d_datalis)):
        d_niobj = scale(d_data.get_fdata(dtype=dtype))
        d_niobj = remove_csf(d_niobj)
        N = d_niobj.shape[-1]
        d_region_arr = np.zeros((n_reg,N))
        for region_index, region_name in enumerate(region_labels):
            reg_mask = atlas_data == region_index+1
            reg_mask*= mask
            d_region_arr[region_index] = np.nanmean(d_niobj[reg_mask],axis = 0) 
        del d_niobj
        for j, method in enumerate(mtd_lis):
           d_arr[j,i] = method(d_region_arr, *args, **kwargs)
        del d_region_arr
        gc.collect()
    p_val = np.empty((len(mtd_lis),n_reg),dtype= dtype)
    t_stat = np.empty((len(mtd_lis),n_reg),dtype= dtype)
    z_score = np.empty((len(mtd_lis),n_reg),dtype= dtype)
    for k in range(len(mtd_lis)):
        t_stat[k],p_val[k] = _perm_test(c_arr[k], d_arr[k],nsim,test_stat)
    z_score = norm.isf(p_val)
    return t_stat,p_val,z_score