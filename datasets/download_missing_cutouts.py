import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as const
import argparse
import pandas as pd
from unagi.unagi import config
from unagi.unagi import hsc
from unagi.unagi.task import hsc_cutout
import downloadCutout
import io
from astropy.io import fits

def get_cutout(ra, dec, id, output_dir):
    
    rect = downloadCutout.Rect.create(
            rerun='pdr3_wide',
            ra=ra, dec=dec,
            sw='5arcsec',
            sh='5arcsec',
            filter='i',
            image=True,
            mask=False,
            variance=False,
        )

    images = downloadCutout.download(rect, user='xczhou', 
                                     password='JQD7x3g7aOg63zsf480Zw/UX2bPuafp6KdtM1R4H')
    
    metadata, data = images[0]
    
    hdus = fits.open(io.BytesIO(data))
    img = hdus[1].data
    
    # Handle each axis separately
    if img.shape[0] > 60:
        img = img[:60, :]
    elif img.shape[0] < 60:
        img = np.pad(img, ((60 - img.shape[0], 0), (0, 0)),
                     mode='constant', constant_values=0)
    
    if img.shape[1] > 60:
        img = img[:, :60] 
    elif img.shape[1] < 60:
        img = np.pad(img, ((0, 0), (60 - img.shape[1], 0)),
                     mode='constant', constant_values=0)
        
    primary_hdu = fits.PrimaryHDU()
    
    primary_hdu.header['RA'] = ra
    primary_hdu.header['DEC'] = dec
    primary_hdu.header['RADIUS'] = 10
    primary_hdu.header['ID'] = id
    
    hdu = fits.ImageHDU(img)
    hdu.header['BAND'] = 'I'
    hdu.header['FROM'] = 'downloadCutout'
    
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(f'{output_dir}/{id}.fits', overwrite=True)

def get_cutout_unagi(ra, dec, output_dir, pdr3):
    
    coord = SkyCoord(ra, dec, unit='deg', frame='icrs')
    ang = 10.0 * u.arcsec
    
    cutout = hsc_cutout(coord, ang, filters='i', 
                        archive=pdr3, output_dir=output_dir,
                        save_output=True, verbose=True)
    
    
if __name__ == '__main__':
    
    df = pd.read_csv(f'matched_sources/desi_spring_matched_selected_coeff.csv')
    
    pdr3 = hsc.Hsc(dr='pdr3', rerun='pdr3_wide', config_file='HSC_DAS/login')
    
    output_dir = f'cutouts/spring_missing'
    os.makedirs(output_dir, exist_ok=True)
    
    missing_ids = np.loadtxt('cutouts/spring_missing_ids.txt', dtype=int)
    
    cutout_ls = os.listdir(output_dir)
    cutout_ids = np.array([int(name.split('.')[0]) for name in cutout_ls])
    
    missing_ids = np.setdiff1d(missing_ids, cutout_ids)
    
    print(len(missing_ids))
    
    for id in missing_ids:
        
        try:
            idx = np.where(df['desi_id'].values == id)[0][0]
            ra = df['desi_ra'].values[idx]
            dec = df['desi_dec'].values[idx]
            get_cutout(ra, dec, id, output_dir)
            print(f'{id} done')
            
        except Exception as e:
            print(f'{id} failed: {e}')
            continue