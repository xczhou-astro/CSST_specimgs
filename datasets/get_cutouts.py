import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
import argparse
from multiprocessing import Pool, cpu_count
from collections import OrderedDict

illustration_count = 0
MAX_ILLUSTRATIONS = 30

class IllustrationCounter:
    def __init__(self, max_illustrations=30):
        self.count = 0
        self.max_illustrations = max_illustrations

    def should_illustrate(self):
        if self.count < self.max_illustrations:
            self.count += 1
            return True
        return False

# Create a single instance at module level
illustration_counter = IllustrationCounter()

def illustrate_cutout(i_cutout, ra, dec, id, z, snr):
    """Create illustration plot for cutout"""
    # Only create illustration if we haven't hit the limit
    if illustration_counter.should_illustrate():
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.imshow(i_cutout.data, cmap='plasma')    
        ax.set_title(f'i band')
        
        ax.text(0.7, 0.9, f'ID={id}', ha='center', va='bottom', transform=ax.transAxes, color='white')
        ax.text(0.7, 0.85, f'ra={ra:.4f} dec={dec:.4f}', ha='center', va='bottom', transform=ax.transAxes, color='white')
        ax.text(0.7, 0.8, f'z={z:.4f}', ha='center', va='bottom', transform=ax.transAxes, color='white')
        ax.text(0.7, 0.75, f'SNR={snr:.4f}', ha='center', va='bottom', transform=ax.transAxes, color='white')

        plt.savefig(f'cutouts/illustrations/{id}.png')
        plt.close()

# Replace the simple cache dictionaries with LRU caches
class LRUCache(OrderedDict):
    def __init__(self, maxsize=10):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)

# Replace the global cache variables
fits_cache = LRUCache(maxsize=100)  # Adjust maxsize based on your memory constraints
wcs_cache = LRUCache(maxsize=100)

def create_hdul(i_cutout, ra, dec, id, z, snr,
                radius, i_fits, out_dir):
    
    i_fits = os.path.basename(i_fits)
    
    # Create a new FITS file with primary HDU and image HDUs
    primary_hdu = fits.PrimaryHDU()
    
    # Add some basic header info to primary HDU
    primary_hdu.header['RA'] = ra
    primary_hdu.header['DEC'] = dec
    primary_hdu.header['RADIUS'] = radius
    primary_hdu.header['ID'] = id
    primary_hdu.header['Z'] = z
    primary_hdu.header['SNR'] = snr
    
    i_hdu = fits.ImageHDU(i_cutout.data)  
    i_hdu.header['BAND'] = 'I'
    i_hdu.header['FROM'] = i_fits
    
    # Combine into HDUList and write to file
    hdul = fits.HDUList([primary_hdu, i_hdu])
    
    hdul.writeto(os.path.join(out_dir, f'{id}.fits'), overwrite=True)
    

def load_fits_data(file_path):
    """Load and cache FITS data and WCS with memory management"""
    fits_data = fits_cache.get(file_path)
    wcs_data = wcs_cache.get(file_path)
    
    if fits_data is None or wcs_data is None:
        with fits.open(file_path) as hdul:
            fits_data = hdul[1].data
            wcs_data = WCS(hdul[1].header)
            fits_cache.put(file_path, fits_data)
            wcs_cache.put(file_path, wcs_data)
    
    return fits_data, wcs_data

def process_single_source(args):
    """Process a single source - used for parallel processing"""
    row, radius, infos, fits_ls, out_dir, existing_ls = args
    id = row['desi_id']
    ra = row['desi_ra']
    dec = row['desi_dec']
    z = row['desi_z']
    snr = row['hsc_i_flux'] / row['hsc_i_flux_err']
    return get_cutouts(id, ra, dec, z, snr, radius, infos, fits_ls, out_dir, existing_ls)

def get_cutouts(id, ra, dec, z, snr, radius, infos, fits_ls, out_dir, existing_ls):
    
    if f'{id}.fits' in existing_ls:
        print(f'{id} already exists')
        return 'existing'
    
    idx = np.where((ra > infos[:, 0]) & (ra < infos[:, 1]) & 
                   (dec > infos[:, 2]) & (dec < infos[:, 3]))[0]
    
    if len(idx) == 0:
        print(f'{id} failed')
        return None
    
    selected_fits = [fits_ls[i] for i in idx]
    i_fits = [file for file in selected_fits if 'I' in file]
    
    position = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    
    if len(i_fits):
        i_fits = i_fits[0]
        
        # Use cached data
        i_data, i_wcs = load_fits_data(i_fits)
        
        try: # not radius, but aperture size (2 * radius)
            i_cutout = Cutout2D(i_data, position, radius * u.arcsec, wcs=i_wcs, mode='strict')
            
            create_hdul(i_cutout, ra, dec, id, z, snr, radius, i_fits, out_dir)
            
            illustrate_cutout(i_cutout, ra, dec, id, z, snr)
            
            print(f'{id} done')
            
            return id
            
        except Exception as e:
            print(f'{id} failed by {e}')
            return None
    
    else:
        print(f'{id} failed no fits')
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=float, default=10)
    parser.add_argument('--nproc', type=int, default=16,
                        help='Number of processes to use (default: number of CPU cores)')
    args = parser.parse_args()
    
    if args.nproc is None:
        args.nproc = cpu_count()
        
    print(f'Using {args.nproc} processes')
    
    os.makedirs('cache', exist_ok=True)
    os.makedirs('cutouts', exist_ok=True)
    os.makedirs('cutouts/illustrations', exist_ok=True)
    
    df = pd.read_csv(f'matched_sources/desi_spring_matched_selected_coeff.csv')
    
    df['desi_id'] = df['desi_id'].astype(str)
    
    fits_ls = os.listdir(f'HSC_DAS/patches')
    fits_ls = [f for f in fits_ls if f.endswith('.fits')]
    fits_ls = [os.path.join(f'HSC_DAS/patches', f) for f in fits_ls]
    
    coord_ranges = np.load(f'HSC_DAS/patch_infos.npz')
    ra_mins = coord_ranges['ra_mins']
    ra_maxs = coord_ranges['ra_maxs']
    dec_mins = coord_ranges['dec_mins']
    dec_maxs = coord_ranges['dec_maxs']
    
    infos = np.column_stack([ra_mins, ra_maxs, dec_mins, dec_maxs])
    
    out_dir = f'cutouts/spring'
    os.makedirs(out_dir, exist_ok=True)
    
    existing_ls = os.listdir(out_dir)
    
    # Prepare arguments for parallel processing
    process_args = [(row, args.radius, infos, fits_ls, out_dir, existing_ls) 
                   for _, row in df.iterrows()]
    
    # Use multiprocessing to process sources in parallel
    with Pool(processes=args.nproc) as pool:
        results = pool.map(process_single_source, process_args)
    
    res_ids = [result for result in results if result is not None]
    total_ids = df['desi_id'].values.tolist()
    missing_ids = [id for id in total_ids if id not in res_ids]
    
    print(f'Missing {len(missing_ids)} cutouts')
    np.savetxt(f'cutouts/spring_missing_ids.txt', missing_ids, fmt='%s')
    
    # Clear caches
    fits_cache.clear()
    wcs_cache.clear()