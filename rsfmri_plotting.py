import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets

def _whole_3d_plot(t_stat, p_val=None,alpha = 0.05, vmin=-3,vmax=3,figsize = (10,7)):
    if p_val is None:
        x,y,z = np.where(np.logical_and(~np.isnan(t_stat),t_stat>0))
    else:
        x, y, z = np.where(p_val < alpha)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=t_stat[x, y, z], cmap='viridis',vmin=vmin, vmax=vmax, s=1)
    plt.title('3D Scatter of significant voxels difference')
    cbar = fig.colorbar(sc,  shrink=0.6, label='t-statistic')
    plt.show()
    
def _overlay(t_stat, p_val = None, overlaymask=None,thres = 2.0, 
            coords = (0, -60, 10), atlas=None):
    reference_img = atlas.maps
    masked_t = t_stat
    if p_val is not None:
        masked_t = np.where(p_val < 0.05, t_stat, np.nan)
    if overlaymask is not None:
        masked_t[~overlaymask]=np.nan
    t_img = nib.Nifti1Image(masked_t.astype(np.float64), reference_img.affine)

    _overlay_plot(t_img, coords=coords, thres=thres)
     
# Plot overlay
def _overlay_plot(test_img, coords=(0, -60, 10),thres = 2.0):
    bg_img = datasets.load_mni152_template()
    plotting.plot_stat_map(
        test_img,
        bg_img=bg_img,
        threshold=thres,  # t-threshold; can be None
        cut_coords=coords,  # MNI coordinates (adjust as needed)
        title='Significant Voxels',
        cmap='coolwarm',
        colorbar=True
    )
    plotting.show()