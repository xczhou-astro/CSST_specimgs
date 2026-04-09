# CSST_specimgs
Spectroscopic redshifts estimated from CSST slitless spectral images  

In this work, we employ Bayesian convolutional neural network (BCNN) to estimate redshifts from mock 2D spectral images for CSST slitless spectroscopic survey. 

## Datasets
This directory includes scripts to generate ideal galaxy images and SEDs for simulation of slitless spectra.  

The ideal images are obtained from HSC-SSP PDR3, and the SEDs are from DESI DR1.

`query.txt`: query command to retrieve sources from [CAS Search](https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/) of HSC-SSP PDR3.  
`downloadCutout.py`: official image cutout python script downloaded from [here](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3/downloadCutout/).  
`downloadPsf.py`: official psf picker python script downloaded fron [here](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3/downloadPsf/).  
`get_cutouts.py`: image cutout script from sky patches downloaded from DAS Search.  
`download_missing_cutouts.py`: get cutouts that are missed by `get_cutouts.py` (edge sources). Call `downloadCutout.py`.  
`psf_downloader.py`: call `downloadPsf.py`.  
`match_with_desi.py`: match sources in HSC and DESI.  
`patch_infos.py`: get the metadata from sky patches downloaded from DAS Search.  
`find_deconvolve_cutouts.py`: mask out the central source for cutouts and perform deconvolution.  
`add_coeff.py`: add 'coeff' column for the selected source catalog from DESI DR1 catalog.  

Other neccessary files:  
DESI redrock templates [`rrtemplate-GALAXY-None-v2.6.fits`](https://github.com/desihub/redrock-templates/blob/main/rrtemplate-GALAXY-None-v2.6.fits);  
DESI DR1 catalog [`zall-pix-iron.fits`](https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits) (20GB).  

## sls
This directory includes scripts to generate mock slitless spectra (2D images and 1D spectra).  

CSST slitless simulation software: [`sls_1d_spec`](https://csst-tb.bao.ac.cn/code/zhangxin/sls_1d_spec).

`sls_generation.py`: generation pipeline for mock slitless spectra.  
`save_lmdb.py`: save the 2D spectral images in lmdb format, facilitate fast data loading by PyTorch.  
`add_properties`: add several fluxes from DESI DR1 catalog.  

## train
This directory includes scripts to train the deterministic pre-training and Bayesian neural networks for redshift estimations

`datasets.py`: data loader for the 2D spectral images.  
`model.py`: neural network architectures.  
`train.py`: training and testing routinue.  

Run training by:
```Python
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train.py
```
or 
```Python
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 train.py
```
