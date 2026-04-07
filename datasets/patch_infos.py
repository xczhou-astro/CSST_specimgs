import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.wcs import WCS
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress WCS warnings for cleaner output
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

def process_patch_file(file_path):
    """Process a single patch file and return its coordinate bounds."""
    try:
        with fits.open(file_path) as hdu:
            header = hdu[1].header
        
        wcs = WCS(header)
        ny, nx = header['NAXIS1'], header['NAXIS2']
        
        # Define corners more efficiently using numpy
        corners = np.array([[0, 0], [0, ny - 1], [nx - 1, 0], [nx - 1, ny - 1]], dtype=np.float64)
        
        # Transform coordinates
        corners_world = wcs.all_pix2world(corners, 0)
        
        ra = corners_world[:, 0]
        dec = corners_world[:, 1]
        
        return np.min(ra), np.max(ra), np.min(dec), np.max(dec)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None

patch_path = 'HSC_DAS/patches'
patch_files = os.listdir(patch_path)
patch_files = [file for file in patch_files if file.endswith('.fits')]

# Pre-allocate arrays for better performance
num_files = len(patch_files)
ra_mins = np.empty(num_files, dtype=np.float64)
ra_maxs = np.empty(num_files, dtype=np.float64)
dec_mins = np.empty(num_files, dtype=np.float64)
dec_maxs = np.empty(num_files, dtype=np.float64)

# Create full file paths for parallel processing
file_paths = [os.path.join(patch_path, file) for file in patch_files]

# Process files in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(process_patch_file, file_paths))

# Extract results into pre-allocated arrays
for i, (ra_min, ra_max, dec_min, dec_max) in enumerate(results):
    if ra_min is not None:  # Check for successful processing
        ra_mins[i] = ra_min
        ra_maxs[i] = ra_max
        dec_mins[i] = dec_min
        dec_maxs[i] = dec_max

# Save results
np.savez('HSC_DAS/patch_infos.npz', 
         ra_mins=ra_mins, 
         ra_maxs=ra_maxs, 
         dec_mins=dec_mins, 
         dec_maxs=dec_maxs)

# Load and plot data efficiently
# df = pd.read_csv('matched_sources/desi_hsc_spring_matched_i_mag_22_snr_100_region_145_165_-1_3_coeff.csv')
# df = pd.read_csv('matched_sources/desi_spring_lt_1arcsec_coeff.csv')
df = pd.read_csv('matched_sources/desi_spring_matched_selected_coeff.csv')

# Create plot with optimized settings
plt.figure(figsize=(10, 10))
plt.scatter(df['desi_ra'], df['desi_dec'], s=1, alpha=0.6, rasterized=True)
plt.scatter(ra_mins, dec_mins, s=1, color='red', alpha=0.8)
plt.scatter(ra_maxs, dec_maxs, s=1, color='red', alpha=0.8)
plt.savefig('HSC_DAS/patch_coverage.png', dpi=150, bbox_inches='tight')
plt.close()  # Free memory