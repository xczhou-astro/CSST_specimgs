import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from astropy.io import fits
import io
import downloadPsf

def get_psf(ra, dec, id=None, output_dir=None):
    rect = downloadPsf.PsfRequest.create(
        rerun='pdr3_wide',
        ra=ra, dec=dec,
        filter='i')
    data = downloadPsf.download(rect, user='xczhou_astro', password='F0hjPTIqPA8g+Xb/S67a78YVy4qAC7SdONkdhXBe')

    metadata, result = data[0]
    hdus = fits.open(io.BytesIO(result))

    primary_hdu = fits.PrimaryHDU(hdus[0].data)
    primary_hdu.header['RA'] = ra
    primary_hdu.header['DEC'] = dec
    primary_hdu.header['ID'] = str(id)
    primary_hdu.header['FILTER'] = 'i'

    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(os.path.join(output_dir, f'{str(id)}.fits'), overwrite=True)
  
if __name__ == '__main__':
    df = pd.read_csv('psfs/remaining.csv')
    df['desi_id'] = df['desi_id'].astype(str)
    
    output_dir = 'psfs/spring_remaining'
    os.makedirs(output_dir, exist_ok=True)
    
    psf_ls = os.listdir(output_dir)
    existing_ids = [str(f.split('.')[0]) for f in psf_ls]
    
    print(f'existing: {len(existing_ids)}')
    
    remaining_ids = np.setdiff1d(df['desi_id'].values, existing_ids)
    print(f'remaining: {len(remaining_ids)}')
    
    df = df[df['desi_id'].isin(remaining_ids)]
    
    print(df.shape)
     
    for i, row in df.iterrows():
        print(i, row['desi_id'])
        get_psf(row['desi_ra'], row['desi_dec'], id=row['desi_id'], output_dir=output_dir)