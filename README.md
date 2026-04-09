# CSST_specimgs
Spectroscopic redshifts estimated from CSST slitless spectral images  

In this work, we employ Bayesian convolutional neural network (BCNN) to estimate redshifts from mock 2D spectral images for CSST slitless spectroscopic survey. 

## Datasets
This directory includes scripts to generate ideal galaxy images and SEDs for simulation of slitless spectra.  

The ideal images are obtained from HSC-SSP PDR3, and the SEDs are from DESI DR1.

`query.txt`: the query command to retrieve sources from [CAS Search](https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/) of HSC-SSP PDR3.  
`downloadCutout.py`: official image cutout python script downloaded from [here](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3/downloadCutout/).  
`downloadPsf.py`: official psf picker python script downloaded fron [here](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3/downloadPsf/).  
`get_cutouts.py`: image cutout script from sky patches downloaded from DAS Search.  
`download_missing_cutouts.py`: get cutouts that are missed by `get_cutouts.py` (edge sources). Call `downloadCutout.py`.  
`psf_downloader.py`: call `downloadPsf.py`.  
`match_with_desi.py`: match sources in HSC and DESI.  
`patch_infos.py`: get the metadata from sky patches downloaded from DAS Search.  
`find_deconvolve_cutouts.py`: mask out the central source for cutouts and perform deconvolution.  
