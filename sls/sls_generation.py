#!/usr/bin/env python3

from SpecGen.SpecGenerator import SpecGenerator
from SpecGen.Config import Config

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import rescale
import galsim
import os
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import warnings
from contextlib import contextmanager

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension

class ParsecDimension(_Dimension):
    def __init__(self):
        super().__init__('pc')
        self.add_units('kpc', 1000)

class AngleDimension(_Dimension):
    def __init__(self):
        super().__init__(r'${^{\prime\prime}}$')


@contextmanager
def timer(description="Task"):
    """Simple context manager to time code execution"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description} completed in {elapsed:.2f} seconds")

def standardize_img(img, xsize, ysize, skybg, dark, readout, expTime, expNum, seed=1234):
    """
    Standardize image dimensions by padding and centering
    Optimized with vectorized operations
    """
    pad_img = padding_img(img, xsize, ysize, skybg, dark, readout, expTime, expNum, seed=seed)
    crop_img = crop_center(pad_img, (ysize, xsize))
    return crop_img

def padding_img(img, xsize, ysize, skybg, dark, readout, expTime, expNum, seed=1234):
    """
    Pad an image to standardized dimensions
    Optimized with vectorized noise generation
    """
    # Ensure dimensions are sufficient
    xsize = max(xsize, img.shape[1])
    ysize = max(ysize, img.shape[0])
    
    # Create container with background noise in one step
    mean = (skybg + dark) * expTime * expNum
    
    # Set random seed once for reproducibility
    np.random.seed(seed)
    
    # Generate container with Poisson noise in a single operation
    container = np.random.poisson(np.full((ysize, xsize), mean))
    
    # Add readout noise for all exposures at once with broadcasting
    for i in range(expNum):
        np.random.seed(seed + i * seed)
        container += np.around(np.random.normal(0, readout, container.shape), 0).astype(int)
    
    # Subtract mean background level - explicitly cast to float to avoid int subtraction issues
    container = container.astype(np.float32) - mean
    
    # Calculate centers for placing the image
    spec_img_height, spec_img_width = img.shape
    start_row = max(0, (ysize - spec_img_height) // 2)
    start_col = max(0, (xsize - spec_img_width) // 2)
    
    # Insert the image into the container
    end_row = min(ysize, start_row + spec_img_height)
    end_col = min(xsize, start_col + spec_img_width)
    
    # Handle potential size mismatch
    img_height_to_use = end_row - start_row
    img_width_to_use = end_col - start_col
    
    container[start_row:end_row, start_col:end_col] = img[:img_height_to_use, :img_width_to_use]
    
    return container

def crop_center(img, crop_size):
    """
    Crop the image to specified dimensions, centered
    Optimized with better bounds checking
    """
    if isinstance(crop_size, int):
        crop_height = crop_width = crop_size
    else:
        crop_height, crop_width = crop_size
    
    height, width = img.shape
    
    # Calculate starting positions with bounds checking
    start_y = max(0, (height - crop_height) // 2)
    start_x = max(0, (width - crop_width) // 2)
    
    # Ensure we don't go out of bounds
    end_y = min(height, start_y + crop_height)
    end_x = min(width, start_x + crop_width)
    
    # Crop the image
    return img[start_y:end_y, start_x:end_x]

def generate_2d_spec(sed, img, config, psf, seed=1234):
    """
    Generate 2D spectral images for an input image
    Returns spectral images, tables, and direct images
    """
    spec_imgs = []
    specTabs = []
    direct_imgs = []
    
    # Set fixed parameters
    skybg = 0.3
    dark = 0.02
    readout = 5
    expTime = 150
    expNum = 4
    
    standard_xsize = 768
    standard_ysize = 128
    
    for i, band in enumerate(['GV', 'GI']):
        # Create spectral generator with fixed seed for reproducibility
        specG = SpecGenerator(sedFn=sed, grating=band, beam='A', aper=2.0,
                             xcenter=2000., ycenter=5000.,
                             p_size=0.074, psf=psf, skybg=skybg,
                             dark=dark, readout=readout, t=expTime, expNum=expNum,
                             config=config, saturation=90000)
        
        # Set seed for reproducibility
        specG.seed = seed + i
        
        # Generate spectral data
        specTab, specImg, direct_img, satPix = specG.generateSpec1dforInputImg(
            img=img, img_pixel_scale=0.074,
            limitfluxratio=0.3, deltLamb=0.01,
            pixel_size=0.074)
        
        # Standardize the spectral image
        specImg = standardize_img(specImg, standard_xsize, standard_ysize,
                                 skybg, dark, readout, expTime, expNum, seed=seed)
        
        # Filter spectral table
        unmasked_idx = np.where(specTab['ERR'] >= 0)[0]
        specTab = specTab[unmasked_idx]
        
        # Collect results
        spec_imgs.append(specImg)
        specTabs.append(specTab)
        direct_imgs.append(direct_img)
    
    return spec_imgs, specTabs, direct_imgs

def get_sed(z, coeff, wave_rf, wave_low, wave_high, temps):
    """
    Get spectral energy distribution (SED) based on coefficients and templates
    Optimized matrix multiplication
    """
    # Convert coefficient string to numpy array
    coeff_array = np.array(coeff.split(',')).astype(np.float32)
    
    # Reshape for matrix multiplication and compute model
    model = np.dot(coeff_array.reshape(1, -1), temps).flatten()
    
    # Apply redshift to wavelength
    wavelength = wave_rf * (1 + z)
    
    # Filter to desired wavelength range
    idx = np.where((wavelength >= wave_low) & (wavelength <= wave_high))[0]
    wavelength = wavelength[idx]
    model = model[idx] * 10**-17
    
    # Stack wavelength and model into a 2D array
    return np.column_stack((wavelength, model))

def get_img(ID, img_size=128, deconvolved_path='../HSC/deconvolved/spring/'):
    """
    Load and preprocess deconvolved image
    With memory mapping and error handling
    """
    try:
        # Use memory mapping for efficiency
        with fits.open(f'{deconvolved_path}{ID}.fits', memmap=True) as hdul:
            img = hdul[1].data.copy()  # Copy to avoid issues after file is closed
            
        # Rescale image to target resolution
        img = rescale(img, 0.168 / 0.074, anti_aliasing=True, mode='reflect')
        
        # Center crop to desired size
        img = crop_center(img, img_size)
        
        # Normalize
        img = img / np.sum(img)
        
        return img
    
    except Exception as e:
        raise RuntimeError(f"Error loading image for ID {ID}: {str(e)}")

def save_spec_imgs(spec_imgs, specTabs, ID, z, ra, dec, info, snr1, snr2, output_dir):
    """
    Save spectral images and data to FITS file
    Optimized file writing
    """
    # Create a primary HDU with metadata
    primary_hdu = fits.PrimaryHDU()
    
    # Add metadata to the primary header
    primary_hdu.header['ID'] = ID
    primary_hdu.header['REDSHIFT'] = z
    primary_hdu.header['RA'] = ra
    primary_hdu.header['DEC'] = dec
    primary_hdu.header['I_MAG'] = info['i_mag']
    primary_hdu.header['R_MAG'] = info['r_mag']
    primary_hdu.header['G_MAG'] = info['g_mag']
    primary_hdu.header['Z_MAG'] = info['z_mag']
    primary_hdu.header['Y_MAG'] = info['y_mag']
    primary_hdu.header['COEFF'] = info['coeff']
    primary_hdu.header['RADIUS'] = info['radius']
    primary_hdu.header['GV_SNR'] = snr1
    primary_hdu.header['GI_SNR'] = snr2
    
    # Create image HDUs for the spectral images
    image_hdu1 = fits.ImageHDU(spec_imgs[0])
    image_hdu2 = fits.ImageHDU(spec_imgs[1])
    
    # Add band information to the image headers
    image_hdu1.header['BAND'] = 'GV'
    image_hdu2.header['BAND'] = 'GI'
    
    # Add specTabs in another hdu
    specTab_hdu1 = fits.BinTableHDU(specTabs[0])
    specTab_hdu1.header['BAND'] = 'GV'
    
    specTab_hdu1.header['SNR'] = snr1
    
    specTab_hdu2 = fits.BinTableHDU(specTabs[1])
    specTab_hdu2.header['BAND'] = 'GI'
    
    specTab_hdu2.header['SNR'] = snr2
    
    # Create HDU list and write to file
    hdul = fits.HDUList([primary_hdu, image_hdu1, image_hdu2, specTab_hdu1, specTab_hdu2])
    
    # Save the FITS file
    output_file = f"{output_dir}/{ID}.fits"
    hdul.writeto(output_file, overwrite=True)

def illustrate_spec_imgs(spec_imgs, specTabs, sed, img, ID, z, snr1, snr2, output_dir):
    """
    Create visualization of spectral images and data with the original galaxy image
    Layout: Left - galaxy image, Right - spectral data (GV, GI bands and spectrum)
    """
    with plt.ioff():  # Non-interactive mode for faster rendering
        # Create figure with two columns
        fig = plt.figure(figsize=(10, 6), dpi=150)
        
        # Set up the grid: 3 rows, 2 columns with different column widths
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])
        
        # Create axes
        # Left column - galaxy image (spans all 3 rows)
        ax_img = fig.add_subplot(gs[:, 0])
        
        # Right column - spectral data (one row each)
        ax_gv = fig.add_subplot(gs[0, 1])
        ax_gi = fig.add_subplot(gs[1, 1])
        ax_spec = fig.add_subplot(gs[2, 1])
        
        angleDim = AngleDimension()
        scalebarSize = 0.25 * img.shape[0] * 0.074
        scalebarUnit = '${^{\prime\prime}}$'
        scalebar = ScaleBar(0.074, '${^{\prime\prime}}$', dimension=angleDim,
                            fixed_value=scalebarSize, fixed_units=scalebarUnit,
                            frameon=False, location='lower right', scale_loc='top',
                            color='white', font_properties={'size': 18})
        ax_img.add_artist(scalebar)
        
        # Plot the galaxy image
        ax_img.imshow(img, origin='lower', cmap='inferno')
        ax_img.set_title(f'Galaxy ID: {ID}\nz: {z:.3f}')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Get dimensions of spectral images
        gv_height, gv_width = spec_imgs[0].shape
        gi_height, gi_width = spec_imgs[1].shape
        
        # Plot GV band spectral image
        ax_gv.imshow(spec_imgs[0], origin='lower', cmap='viridis', 
                   extent=[0, gv_width, 0, gv_height], aspect='auto')
        ax_gv.set_title('GV Band')
        ax_gv.set_xticks([])
        ax_gv.set_yticks([])
        
        # Plot GI band spectral image
        ax_gi.imshow(spec_imgs[1], origin='lower', cmap='viridis', 
                   extent=[0, gi_width, 0, gi_height], aspect='auto')
        ax_gi.set_title('GI Band')
        ax_gi.set_xticks([])
        ax_gi.set_yticks([])
        
        # Plot spectral data
        ax_spec.plot(specTabs[0]['WAVELENGTH'], specTabs[0]['FLUX'], 'b-', label='GV')
        ax_spec.plot(specTabs[1]['WAVELENGTH'], specTabs[1]['FLUX'], 'r-', label='GI')
        ax_spec.plot(sed[:, 0], sed[:, 1], 'k--', label='SED')
        
        # Set axis limits and labels for spectrum plot
        ax_spec.set_xlim(2500, 10000)
        ax_spec.set_ylim(0, None)
        ax_spec.set_xlabel('Wavelength (Å)')
        ax_spec.set_ylabel('Flux')
        ax_spec.legend(fontsize=8, frameon=False, loc='upper right')
        
        # Add SNR information if available

        snr_text = f"SNR: GV={snr1:.1f}, GI={snr2:.1f}"
        ax_spec.text(0.02, 0.98, snr_text, transform=ax_spec.transAxes,
                   fontsize=8, verticalalignment='top')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = f"{output_dir}/{ID}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', transparent=False)
        
        plt.close(fig)

def process_single_object(row, config, psf, wave_rf, wave_low, wave_high, temps, 
                        output_dir, vis_dir, deconvolved_path, num_to_visualize):
    """Process a single object (for parallel execution)"""
    ID = row['desi_id']
    z = row['desi_z']
    ra = row['desi_ra']
    dec = row['desi_dec']
    i_mag = row['hsc_i_mag']
    i_flux = row['hsc_i_flux']
    i_flux_err = row['hsc_i_flux_err']
    r_mag = row['hsc_r_mag']
    r_flux = row['hsc_r_flux']
    r_flux_err = row['hsc_r_flux_err']
    g_mag = row['hsc_g_mag']
    g_flux = row['hsc_g_flux']
    g_flux_err = row['hsc_g_flux_err']
    z_mag = row['hsc_z_mag']
    z_flux = row['hsc_z_flux']
    z_flux_err = row['hsc_z_flux_err']
    coeff = row['coeff']
    y_mag = row['hsc_y_mag']
    radius = row['desi_shape_r']
    count = row.name  # Get the row index
    
    info = {}
    info['i_mag'] = i_mag
    info['r_mag'] = r_mag
    info['g_mag'] = g_mag
    info['z_mag'] = z_mag
    info['y_mag'] = y_mag
    info['i_flux'] = i_flux
    info['i_flux_err'] = i_flux_err
    info['r_flux'] = r_flux
    info['r_flux_err'] = r_flux_err
    info['g_flux'] = g_flux
    info['g_flux_err'] = g_flux_err
    info['z_flux'] = z_flux
    info['z_flux_err'] = z_flux_err
    info['coeff'] = coeff
    info['radius'] = radius
    
    for key in info:
        if pd.isna(info[key]):
            info[key] = -99
    
    try:
        # Get SED
        sed = get_sed(z, coeff, wave_rf, wave_low, wave_high, temps)
        
        # Get image
        img = get_img(ID, img_size=128, deconvolved_path=deconvolved_path)
        
        # Generate 2D spectral data
        spec_imgs, specTabs, direct_imgs = generate_2d_spec(sed, img, config, psf, seed=count)
        
        # Calculate SNR
        gv_flux = specTabs[0]['FLUX'].astype(np.float32)
        gv_err = specTabs[0]['ERR'].astype(np.float32)
        
        idx_mask1 = (gv_flux < 0) | (gv_err <= 0)
        gv_flux[idx_mask1] = 0
        gv_err[idx_mask1] = -1
        
        snr1 = np.mean(gv_flux / gv_err)
        
        gi_flux = specTabs[1]['FLUX'].astype(np.float32)
        gi_err = specTabs[1]['ERR'].astype(np.float32)
        
        idx_mask2 = (gi_flux < 0) | (gi_err <= 0)
        gi_flux[idx_mask2] = 0
        gi_err[idx_mask2] = -1
        
        snr2 = np.mean(gi_flux / gi_err)
        # Save spectrum data
        save_spec_imgs(spec_imgs, specTabs, ID, z, ra, dec, info, snr1, snr2, output_dir)
        
        snrs = {
            'snr1': snr1,
            'snr2': snr2,
        }
        
        # Generate visualization if needed
        if count < num_to_visualize:
            # print('visualizing')
            illustrate_spec_imgs(spec_imgs, specTabs, sed, img, ID, z, snr1, snr2, vis_dir)
            # print('visualized')
        
        return ID, True, None, snrs
    
    except Exception as e:
        print(f"Error processing object {ID}: {str(e)}")
        return ID, False, str(e), {}

def main():
    # Start timing
    start_time = time.time()
    
    # Suppress common warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print("Starting accelerated SLS generation...")
    
    # Load data
    with timer("Loading input data"):
        df = pd.read_csv('../datasets/matched_sources/desi_spring_matched_selected_coeff.csv')
        template = fits.open('../datasets/DESI/rrtemplate-GALAXY-None-v2.6.fits', memmap=True)
        temps = template[0].data  # (10, 97720)
        
        wave_rf = template[0].header['CRVAL1'] + template[0].header['CDELT1'] * np.arange(temps.shape[1])
        
        wave_low = 2000  # obs frame
        wave_high = 10000  # obs frame
        
        dataDir = '../sls_1d_spec/data/'
        config = Config(dataDir=dataDir)
        psf = galsim.Gaussian(fwhm=0.3)
    
    # Create output directories
    output_dir = 'output'
    sls_specs = 'output/sls_specs'
    sls_vis = 'output/sls_vis'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sls_specs, exist_ok=True)
    os.makedirs(sls_vis, exist_ok=True)
    
    # Clear failed IDs file
    with open('output/failed_ids.txt', 'w') as f:
        pass
    
    num_workers = 16
    
    # For visualization
    num_to_visualize = 30
    
    deconvolved_path = '../datasets/deconvolved/spring/'
    deconvolved_files = os.listdir(deconvolved_path)
    deconvolved_ids = [int(file.split('.')[0]) for file in deconvolved_files]
    
    exist_ls = os.listdir(sls_specs)
    exist_ids = [int(f.split('.')[0]) for f in exist_ls]
    
    remaining_ids = np.setdiff1d(deconvolved_ids, exist_ids)
    
    print('Existing IDs:', len(exist_ids))
    print('Remaining IDs:', len(remaining_ids))

    deconvolved_ids = remaining_ids

    df = df[df['desi_id'].isin(deconvolved_ids)]
    df.drop_duplicates(subset=['desi_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Processing {len(df)} objects using {num_workers} workers...")
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_single_object,
        config=config,
        psf=psf,
        wave_rf=wave_rf,
        wave_low=wave_low,
        wave_high=wave_high,
        temps=temps,
        output_dir=sls_specs,
        vis_dir=sls_vis,
        deconvolved_path=deconvolved_path,
        num_to_visualize=num_to_visualize
    )
    
    # Process in parallel
    failed_ids = []
    successful_ids = []
    
    # Collect SNRs keyed by ID so ordering can't mismatch
    snr_rows = []
    
    with timer("Processing all objects"):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all rows to the executor
            futures = {executor.submit(process_func, row): idx for idx, row in df.iterrows()}
            
            # Process results as they complete
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing objects"):
                ID, success, error, snrs = future.result()
                if success:
                    successful_ids.append(ID)
                    snr_rows.append({
                        'desi_id': int(ID),
                        'gv_snr': float(snrs['snr1']),
                        'gi_snr': float(snrs['snr2']),
                    })
                else:
                    failed_ids.append(f"{ID} failed: {error}")
                    with open('output/failed_ids.txt', 'a') as f:
                        f.write(f"{ID} failed: {error}\n")
    
    successful_ids = np.array(successful_ids).astype(int)
    
    # Join SNRs back by ID (do NOT rely on positional alignment)
    df_successful = df[df['desi_id'].isin(successful_ids)].copy()
    snr_df = pd.DataFrame(snr_rows)
    df_successful = df_successful.merge(snr_df, on='desi_id', how='inner', validate='one_to_one')
    df_successful.to_csv('output/sls_catalogue.csv', index=False)
    
    # Write summary statistics
    elapsed = time.time() - start_time
    with open('output/summary.txt', 'w') as f:
        f.write(f"Total processing time: {elapsed:.2f} seconds\n")
        f.write(f"Objects processed: {len(df)}\n")
        f.write(f"Successful: {len(successful_ids)}\n")
        f.write(f"Failed: {len(failed_ids)}\n")
        
    print(f"Completed processing {len(successful_ids)} objects successfully")
    print(f"Failed: {len(failed_ids)} objects")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average time per object: {elapsed/len(df):.2f} seconds")

if __name__ == "__main__":
    main() 