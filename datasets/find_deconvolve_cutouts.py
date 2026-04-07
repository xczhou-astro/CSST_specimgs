import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from skimage.restoration import richardson_lucy
import pandas as pd
from astropy.io import fits
from photutils.segmentation import detect_threshold, SourceCatalog, SourceFinder
import tqdm
import matplotlib.pyplot as plt

COUNT = 0

# Try to import cupy for GPU acceleration

def find_center_source(img, nsigma=5.0):
    """
    Find and isolate the central source in an image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        The input image.
    nsigma : float, optional
        The number of standard deviations above background for threshold detection.
        
    Returns:
    --------
    tuple
        (masked_image, nsigma, nsources) or (None, nsigma, nsources) if no sources found
    """
    # Use a try-except to handle images with problematic content
    try:
        threshold = detect_threshold(img, nsigma=nsigma)
        finder = SourceFinder(npixels=9, progress_bar=False)
        segm = finder(img, threshold)
        
        if segm is None or segm.nlabels == 0:
            return None, nsigma, 0, 'no sources found'
            
        catalog = SourceCatalog(img, segm)
        
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        
        # Vectorized distance calculation
        xcentroids = np.array([source.xcentroid for source in catalog])
        ycentroids = np.array([source.ycentroid for source in catalog])
        distances = np.sqrt((xcentroids - cx)**2 + (ycentroids - cy)**2)
        
        idx = np.argmin(distances)
        
        # centerx = catalog[idx].xcentroid
        # centery = catalog[idx].ycentroid
        
        # Create mask for the center segment
        center_segment_mask = segm.data == (idx + 1)
        
        # check if the central pixel is part of the center segment
        if not center_segment_mask[cy, cx]:
            return None, nsigma, len(catalog), 'central pixel not part of center segment'
        
        masked_center_img = img * center_segment_mask
        
        # Check if the maximum of masked_center_img is within a 3-pixel radius of the center
        max_value = np.max(masked_center_img)
        if max_value > 0:  # Ensure there's a valid maximum
            # Find coordinates of the maximum value
            max_coords = np.where(masked_center_img == max_value)
            # If there are multiple pixels with the same maximum value, take the first one
            max_y, max_x = max_coords[0][0], max_coords[1][0]
            
            # Calculate distance from center
            distance_from_center = np.sqrt((max_x - cx)**2 + (max_y - cy)**2)
            
            # Check if maximum is within 3 pixels of center
            if distance_from_center > 3:
                return None, nsigma, len(catalog), 'maximum not within 3 pixels of center'
        
        # shift_x = int(cx - centerx)
        # shift_y = int(cy - centery)
        
        # # Optimize shift operation
        # shifted_img = np.roll(masked_center_img, (shift_y, shift_x), axis=(0, 1))
        # valid_mask = np.roll(center_segment_mask, (shift_y, shift_x), axis=(0, 1))
        
        # shifted_img[~valid_mask] = 0
        
        return masked_center_img, nsigma, len(catalog), 'success'
    
    except Exception as e:
        print(f"Error in find_center_source: {e}")
        return None, nsigma, 0, 'error'

def deconvolve_cpu(img, psf, max_iter=30, clip=False, filter_epsilon=1e-2):
    """CPU-based Richardson-Lucy deconvolution"""
    img = img / np.max(img)
    img = np.clip(img, 0, 1)
    psf = psf / np.max(psf)
    return richardson_lucy(img, psf, num_iter=max_iter, clip=clip, filter_epsilon=filter_epsilon) 
 
def central_crop(image, crop_size):
    
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size
        
    h, w = image.shape
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    
    return image[start_y: start_y + crop_h, start_x: start_x + crop_w]

def process_file(id, cutouts_ids, total_cutouts, psfs_ids, psfs, denoise_model, output_path):
    """Process a single file"""
    try:
        idx_cutout = cutouts_ids.index(id)
        idx_psf = psfs_ids.index(id)
        
        cutout_path = total_cutouts[idx_cutout]
        psf_path = psfs[idx_psf]
        
        # Open FITS files
        with fits.open(cutout_path) as cutout, fits.open(psf_path) as psf:
            cutout_data = cutout[1].data # 60 
            psf_data = psf[0].data
            
        
        # # first deconvolve and them find sources
        # deconvolved = deconvolve_cpu(cutout_data, psf_data, clip=False, filter_epsilon=0.1)
        # deconvolved = deconvolved / np.max(deconvolved)
        # img, nsigma, nsources, status = find_center_source(deconvolved, nsigma=5.0)
        # final_img = img
        
        # if img is None:
        #     return id, 1, status

        # Find source and then deconvolve
        img, nsigma, nsources, status = find_center_source(cutout_data, nsigma=5.0)
        
        if img is None:
            return id, 1, status  # Failed
        
        # Apply deconvolution
        deconvolved = deconvolve_cpu(img, psf_data, clip=False, filter_epsilon=0.1)
        deconvolved = deconvolved / np.max(deconvolved)
        deconvolved = np.clip(deconvolved, 0, 1)
        final_img = deconvolved
        
        
        # Create output FITS file
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['ID'] = id
        primary_hdu.header['NSOURCES'] = nsources
        primary_hdu.header['NSIGMA'] = nsigma
        
        hdu = fits.ImageHDU(final_img)
        hdul = fits.HDUList([primary_hdu, hdu])
        
        output_file = os.path.join(output_path, f'{id}.fits')
        hdul.writeto(output_file, overwrite=True)
        
        global COUNT
        COUNT += 1
        
        if COUNT < 30:
            plt.figure(figsize=(6, 6))
            plt.imshow(final_img, cmap='gray')
            plt.title(f'ID: {id}')
            plt.axis('off')
            plt.savefig(f'deconvolved/illustrations/{id}.png')
            plt.close()
        
        return id, 0, status  # Success
    
    except Exception as e:
        print(f'{id} failed: {e}')
        return id, 1, 'error'  # Failed

def batch_process_files(ids_batch, cutouts_ids, total_cutouts, psfs_ids, psfs, denoise_model, output_path):
    """Process a batch of files"""
    results = []
    for id in ids_batch:
        result = process_file(id, cutouts_ids, total_cutouts, psfs_ids, psfs, denoise_model, output_path)
        results.append(result)
    return results

if __name__ == '__main__':
    start_time = time.time()
    
    # Define paths
    cutouts_path = 'cutouts/spring'
    cutouts_missing_path = 'cutouts/spring_missing'
    psfs_path = 'psfs/spring'
    psfs_remaining_path = 'psfs/spring_remaining'
    output_path = 'deconvolved/spring'
    illustration_path = 'deconvolved/illustrations'
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(illustration_path, exist_ok=True)
    
    print("Loading file lists...")
    cutouts_ls = os.listdir(cutouts_path)
    cutouts_ids = [cutout.split('.')[0] for cutout in cutouts_ls]
    
    cutouts_ls_missing = os.listdir(cutouts_missing_path)
    cutouts_ids_missing = [cutout.split('.')[0] for cutout in cutouts_ls_missing]
    
    cutouts_ids.extend(cutouts_ids_missing)
    
    total_cutouts = [os.path.join(cutouts_path, cutout) for cutout in cutouts_ls]
    total_cutouts += [os.path.join(cutouts_missing_path, cutout) for cutout in cutouts_ls_missing]
    
    psfs_ls = os.listdir(psfs_path)
    psfs_ids = [psf.split('.')[0] for psf in psfs_ls]
    
    psfs_remaining_ls = os.listdir(psfs_remaining_path)
    psfs_remaining_ids = [psf.split('.')[0] for psf in psfs_remaining_ls]
    
    psfs_ids.extend(psfs_remaining_ids)
    
    psfs = [os.path.join(psfs_path, psf) for psf in psfs_ls]
    psfs += [os.path.join(psfs_remaining_path, psf) for psf in psfs_remaining_ls]
    
    shared_ids = np.intersect1d(cutouts_ids, psfs_ids)
    
    print('Shared IDs: ', len(shared_ids))

    num_workers = 16
    
    batch_size = max(1, min(100, len(shared_ids) // num_workers))
    
    # Create batches
    batches = [shared_ids[i:i + batch_size] for i in range(0, len(shared_ids), batch_size)]
    
    failed_ids = []
    
    print(f"Processing with {num_workers} workers in {len(batches)} batches")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a partial function with all arguments except the batch
        process_batch = partial(batch_process_files, 
                               cutouts_ids=cutouts_ids, 
                               total_cutouts=total_cutouts,
                               psfs_ids=psfs_ids, 
                               psfs=psfs,
                               denoise_model=None,
                               output_path=output_path)
        
        # Submit all batches to the executor
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        with open('deconvolved/failed_ids.txt', 'w') as f:
        
            # Process results as they complete
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                for id, flag, status in future.result():
                    if flag == 1:
                        info = f'{id} failed: {status}'
                        f.write(info + '\n')
                        failed_ids.append(id)
                    else:
                        pass
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Successfully processed {len(shared_ids) - len(failed_ids)} files")
    print(f"Failed to process {len(failed_ids)} files")