# rsfMRI_RLA
rsfMRI_Region-level_Analysis

## Dependencies
WIP
The required dependencies to use the software are listed in the file .

## Getting Started
### Importing class
```{bash}
from Draft2 import *
```

### Loading data
The control and patient data can be loaded into the class `temp_rsfmri` through the initialisation. Using this method requires both the control and patient data to be a list of `nibabel.nifti1.Nifti1Image`. 
```{bash}
test = temp_rsfmri(control_data,disease_data)
```
### Default Atlas & standard_mask
A default atlas is initialised for anatomical region-level analysis. It is also used to create a mask of valid voxels for analysis, where voxels outside of the atlas will be excluded. In additon, it also provide the reference affine for any plotting. The default atlas is the Harvard Oxford Cortical Structural Area. It is loaded from the nilearn dataset: `nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')`. It has a shape of 91 x 109 x 91. Do change the `temp_rsfmri.atlas` if your data is of a different shape. You need to also change `temp_rsfmri.standard_mask`. If you require a different masking, you can also change `temp_rsfmri.standard_mask` without changing the atlas. `temp_rsfmri.standard_mask` is a boolean numpy array where the value of the voxels to be excluded is `False`.

There is also an option to add a additional masking ontop of the standard_mask. `temp_rsfmri.add_data_mask` loads a `nibabel.nifti1.Nifti1Image` from the mask_path parameter, and do a AND operation between the standard_mask and the additional mask.

### Dataprocessing: Removing CSF
This library can only do addtional data processing such as removing CSF. Any other fMRI pre-processing required should be done before the data loading, hence the data loaded should be cleaned of noise from MRI, slice acquistion, motion etc. Currently, 2 dataprocessing method involving CSF are used. The CSF reference used for this dataprocessing comes from the `temp_rsfmri.csf_voxel`. The default voxels used are from the left and right ventricle at `(50, 55, 47), (39, 55, 47)` in voxel space, which is also `(-10, -16, 22), (12, -16, 22)` in MNI space

1. Scaling using CSF: This scales the data to a reference value using the mean of CSF signal within the CSF voxels. 
2. Regression of CSF: This remove the CSF noise by linear regressing the mean CSF signal out of all voxel data.

If you do not require any of these methods, use `temp_rsfmri.fmri.no_modify`.

### Statistical Measures
These methods characterise the fMRI signal using their relevant statisitcal measure. While they are named as voxel methods, they can be used for any data. They operate using array operations at `axis = -1`. Currently, there are 7 statisitcal methods.

1. Mean
2. Standard Deviation
3. Amplitude of Low Frequncy Fluctuations (ALFF)
4. fractional ALFF (fALFF)
5. Spectral Centriod
6. Spectral Variance
7. Spectral Skewness

### Statisitcal Hypothesis Testing
This library curretly supports 2 types of analysis: voxel-level and anatomical region-level. For both analysis, the default test statistic used is Welch's *t*-test. Permutation testing is used for the null distribution instead of the Student's *t*-test. The default multipe testing correction is done using Benjamini-Hochberg procedure as the False Discovery Rate (FDR) correction. The test function returns a test-statistical map, *p*-value map, and a z-score map. The z-score is generated from the *p*-value map using inverse Cumulative Distribution Function (CDF) of the standard normal distribution. The z-score map is created to allow for easier comparision as the test-statisitcal map uses permutation testing, which result using different null distribution for each test. 

#### Voxel level analysis
Testing at voxel-level:
```{bash}
voxel_perm_test(statistical_method)
```
FDR correction using all valid voxels in the brain volume defined by the atlas:
```{bash}
zmap_thres(z_score_map)
```
FDR correction by regions from an atlas:
```{bash}
region_correction(p_val_map, atlas)
```
Testing at anatomical region-level, this function uses a list of statisical methods, allowing for more than 1 analysis in a single run:
```{bash}
region_perm_test(statistical_method_list, atlas)
```
To do correction for anatomical region-level analysis, define the `control` parameter. The list of valid controls are the same as the methods from `statsmodels.stats.multitest.multipletests`.


