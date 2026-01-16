import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns

from tqdm import tqdm
from scipy.stats import ttest_ind,chisquare,f,ncf,norm
from scipy.fft import fft, fftfreq
from statsmodels.stats.multitest import multipletests
from nilearn import datasets, plotting
from nilearn.image import math_img, threshold_img
from nilearn.glm import threshold_stats_img

from mpl_toolkits.mplot3d import Axes3D

from rsfmri_dataprocess import (_no_modify,_regress_csf,_rem_scale, _csf_mean)
from rsfmri_method import (_voxel_mean, _voxel_sd, _voxel_alff, _voxel_falff,
                           _voxel_spec_cent, _voxel_spec_var, _voxel_spec_skewness)
from rsfmri_statstest import (_voxel_perm_test,_welch_ttest,_zmap_thres,
                              _region_correction, _pval_percent, _region_perm_test)
from rsfmri_plotting import (_whole_3d_plot,_overlay)


class rsfmri_RLA():
    
    def __init__(self, c_datalis = None, d_datalis =  None):
        self.shape = (91,109,91)
        self.atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        self.standard_mask = self.atlas.maps.dataobj != 0
        self.tr = 2.2
        self.csf_voxel = [(50, 55, 47),(39, 55, 47)]
        self.control_datalis = c_datalis
        self.disease_datalis = d_datalis
        self.dtype = np.float32
        
    def loadfrom_folder(filepath):
        return None
        
    def data_mask(self,ratio = 0.7):
        # ratio is the percentage of presence in population sample. 
            # e.g. 70% of each sample has to have data to be included in mask
        # shape = c_datalis[0].dataobj.shape[:-1]  # Change self if needed
        n_c = len(self.control_datalis)
        c_mask = np.full(self.shape, 0)
        n_d = len(self.disease_datalis)
        d_mask = np.full(self.shape, 0)
        for c_data in tqdm(self.control_datalis):
            c_mask += (np.abs(np.mean(c_data.dataobj,axis = -1))>0).astype(int)
        for d_data in tqdm(self.disease_datalis):
            d_mask += (np.abs(np.mean(d_data.dataobj,axis = -1))>0).astype(int)
        data_mas = (c_mask>(n_c*ratio))*(d_mask>(n_d*ratio))
        return data_mas
    
    def add_data_mask(self,mask_file):  ### add at2 first####
        try:
            data_mask1 = nib.load(mask_file).get_fdata().astype('bool')
        except:
            data_mask1 = self.data_mask()
            t_img = nib.Nifti1Image(data_mask1.astype(np.float64), self.atlas.maps.affine)
            nib.save(t_img, mask_file)
        self.standard_mask *= data_mask1
        
    def load_atlas(self, atlas_name):
        return datasets.fetch_atlas_harvard_oxford(atlas_name)
        
############## Data processing from rsfmri_dataprocess.py ###########################
    def no_modify(self, data, *args, **kwargs):
        return _no_modify(data, *args, **kwargs)
        
    def csf_mean(self, data):
        return _csf_mean(data,self.csf_voxel)
    
    def regress_csf(self, data, csf_method = None):
        csf_method = self.csf_mean if csf_method is None else csf_method
        return _regress_csf(data, csf_method)

    def remove_scale(self, data, scale = 10_000,csf_method = None):
        csf_method = self.csf_mean if csf_method is None else csf_method
        return _rem_scale(data, scale, self.csf_mean)
    
    
############## Statistical Measure from rsfmri_methods.py ###########################
    def voxel_mean(self,data):
        return _voxel_mean(data)

    def voxel_sd(self,data,ddof=1):
        return _voxel_sd(data,ddof)

    # Total time=Number of timepoints×Repetition Time (TR)
    def voxel_alff(self, data,low=0.01, high=0.08):
        return _voxel_alff(data, self.standard_mask, self.tr, low, high)

    def voxel_falff(self, data, low=0.01, high=0.08):
        return _voxel_falff(data,self.tr,low,high)

    def voxel_spec_cent(self, data, low=0.01, high=0.08):
        return _voxel_spec_cent(data,self.tr,low,high)

    def voxel_spec_var(self, data, low=0.01, high=0.08):
        return _voxel_spec_var(data,self.tr,low,high)

    def voxel_spec_skewness(self, data, low=0.01, high=0.08):
        return _voxel_spec_skewness(data,self.tr,low,high)
    
############## Statistical testing from rsfmri_statstest .py ###########################   
    def welch_ttest(self,x,y):
        return _welch_ttest(x, y)
    
    def voxel_perm_test(self, method, remove_csf = None, scale = None, nsim = 1000, 
                         test_stat = None, *args, **kwargs):
        
        remove_csf = self.regress_csf if remove_csf is None else remove_csf
        scale = self.remove_scale if scale is None else scale
        test_stat = self.welch_ttest if test_stat is None else test_stat
        t_stat, p_val, z_score = _voxel_perm_test(method, self.control_datalis, self.disease_datalis, 
                                                  remove_csf, scale, self.standard_mask, self.shape, 
                                                  self.dtype, nsim, test_stat, *args, **kwargs)
        return t_stat, p_val, z_score
    
    def whole_brain_fdr(self, p_val, alpha = 0.05, cluster = 10, two_sided = False,
                   height_control = 'fdr', plot = False, vmin = 0, vmax = 3):
        z_map = norm.isf(p_val)
        z_corrected = _zmap_thres(z_map, alpha, cluster, self.dtype,
                                  two_sided, height_control, self.atlas)
        
        if plot:
            self.whole_3d_plot(z_corrected, vmin = vmin, vmax = vmax)
        p_corrected = norm.sf(z_corrected)
        return p_corrected, z_corrected
    
    def region_correction(self, p_val, atlas, control = 'fdr_bh', alpha = 0.05, re_dic = False, cluster= None):
        return _region_correction(p_val, atlas, control = 'fdr_bh', alpha = 0.05, re_dic = False,
                                  standard_mask = self.standard_mask,cluster = cluster)
    
    def pval_percent(self,p_val,atlas = None):
        return _pval_percent(p_val,atlas, standard_mask = self.standard_mask)
    
    def region_perm_test(self, mtd_lis, atlas, remove_csf = None, scale = None,
                          nsim = 5_000, test_stat = None, control = None, *args, **kwargs):
        remove_csf = self.regress_csf if remove_csf is None else remove_csf
        scale = self.remove_scale if scale is None else scale
        test_stat = self.welch_ttest if test_stat is None else test_stat
        t_stat, p_val, z_score = _region_perm_test(mtd_lis, atlas, self.control_datalis, self.disease_datalis,
                                                  remove_csf, scale, self.standard_mask, self.dtype, 
                                                  nsim, test_stat, *args, **kwargs)
        if control is not None:
            p_val = multipletests(p_val,method=control)[1]
            z_score = norm.isf(p_val)
        return t_stat, p_val, z_score
        
    
############## Statistical testing from rsfmri_plotting.py ########################### 
    def whole_3d_plot(self, t_stat, p_val=None,alpha = 0.05, vmin=-3,vmax=3,figsize = (10,7)):
        _whole_3d_plot(t_stat, p_val, alpha, vmin, vmax, figsize)
        
    def overlay(self, t_stat, p_val = None, thres = 2.0, coords = (0, -60, 10)):
        _overlay(t_stat, p_val , overlaymask = self.standard_mask, thres = thres, 
                    coords = coords, atlas=self.atlas)
        