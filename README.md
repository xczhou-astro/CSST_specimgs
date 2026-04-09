# CSST_specimgs

Spectroscopic redshift estimation from CSST slitless spectral images.

This project uses a Bayesian convolutional neural network (BCNN) to estimate redshifts from mock 2D spectral images for the CSST slitless spectroscopic survey.

## datasets

This directory contains scripts for generating ideal galaxy images and SEDs used in slitless spectroscopy simulations.

The ideal images are obtained from HSC-SSP PDR3, and the SEDs are taken from DESI DR1.

`query.txt`: Query command used to retrieve sources from [HSC-SSP PDR3 CAS Search](https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/).  
`downloadCutout.py`: Official Python script for image cutout downloads, obtained from [here](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3/downloadCutout/).  
`downloadPsf.py`: Official Python script for PSF downloads, obtained from [here](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3/downloadPsf/).  
`get_cutouts.py`: Gets image cutouts from sky patches downloaded via DAS Search of HSC.  
`download_missing_cutouts.py`: Retrieves cutouts missed by `get_cutouts.py` (typically edge sources) by calling `downloadCutout.py`.  
`psf_downloader.py`: Wrapper script that calls `downloadPsf.py`.  
`match_with_desi.py`: Matches sources between HSC and DESI catalogs.  
`patch_infos.py`: Extracts metadata from sky patches downloaded via DAS Search of HSC.  
`find_deconvolve_cutouts.py`: Masks out central sources in cutouts and performs deconvolution.  
`add_coeff.py`: Adds a `coeff` column to the selected source catalog from DESI DR1.  

Other necessary files:  
DESI redrock template [`rrtemplate-GALAXY-None-v2.6.fits`](https://github.com/desihub/redrock-templates/blob/main/rrtemplate-GALAXY-None-v2.6.fits);  
DESI DR1 catalog [`zall-pix-iron.fits`](https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits) (~20 GB).  

## sls

This directory contains scripts for generating mock slitless spectra (2D images and 1D spectra).

CSST slitless simulation software: [`sls_1d_spec`](https://csst-tb.bao.ac.cn/code/zhangxin/sls_1d_spec).

`sls_generation.py`: End-to-end generation pipeline for mock slitless spectra.  
`save_lmdb.py`: Stores 2D spectral images in LMDB format to support fast PyTorch data loading.  
`add_properties.py`: Adds more flux-related properties from the DESI DR1 catalog.  

## train

This directory contains scripts for deterministic pretraining and Bayesian neural network training for redshift estimation.

`datasets.py`: Data loader for 2D spectral images.  
`model.py`: Neural network architectures.  
`train.py`: Training and evaluation routine.  

Run training with:
```python
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train.py
```
or:
```python
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 train.py
```
