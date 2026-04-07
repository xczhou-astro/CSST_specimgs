import os
from astropy.io import fits
import lmdb
import numpy as np
import pickle
import csv
from multiprocessing import Pool, cpu_count
from functools import partial

GI_SNR_threshold = 1

def crop_center(img, crop_size=(40, 480)):
    
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

def read_fits(fits_path, shape=(128, 768), crop_size=(40, 480)):
    
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        # gv_header = hdul[1].header
        gv_img = hdul[1].data
        
        # gi_header = hdul[2].header
        gi_img = hdul[2].data
        
    assert gv_img.shape == shape and gi_img.shape == shape
    
    keys = ['ID', 'REDSHIFT', 'RA', 'DEC', 'I_MAG', 'R_MAG', 
            'G_MAG', 'Z_MAG', 'Y_MAG', 'COEFF', 'RADIUS', 'GV_SNR', 'GI_SNR']
    
    infos = {}
    for key in keys:
        infos[key] = header[key]
    
    # Crop images to reduce data size
    gv_img = crop_center(gv_img, crop_size)
    gi_img = crop_center(gi_img, crop_size)
    
    # Stack images to shape (2, 40, 480)
    stacked_img = np.stack([gv_img, gi_img], axis=0)
        
    return stacked_img, infos

def process_single_fits(fits_path, shape=(128, 768), crop_size=(40, 480)):
    """Process a single FITS file and return data if GI_SNR passes threshold."""
    try:
        stacked_img, infos = read_fits(fits_path, shape=shape, crop_size=crop_size)
        
        # Add filename to infos for tracking
        infos['filename'] = os.path.basename(fits_path)
        
        # Check GI_SNR threshold
        gi_snr = infos['GI_SNR']
        if gi_snr < GI_SNR_threshold:
            return None, infos, False  # Below threshold
        
        return stacked_img, infos, True  # Above threshold
        
    except Exception as e:
        print(f"  Error processing {fits_path}: {e}")
        return None, None, None


if __name__ == '__main__':
    
    sls_specs = 'output/sls_specs'
    sls_ls = os.listdir(sls_specs)
    sls_ls = [os.path.join(sls_specs, f) for f in sls_ls if f.endswith('.fits')]
    
    height = 128
    width = 768
    crop_size = (40, 480)
    
    # Configuration
    data_per_file = 10000
    output_dir = 'output/lmdb_data'
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = len(sls_ls)
    print(f"Total FITS files: {total_files}")
    print(f"GI_SNR threshold: {GI_SNR_threshold}")
    print(f"Crop size: {crop_size}")
    
    cores = 16
    print(f"Using {cores} CPU cores for parallel processing")
    
    # Process all files with multiprocessing
    print("\nProcessing FITS files...")
    process_func = partial(process_single_fits, shape=(height, width), crop_size=crop_size)
    
    with Pool(processes=cores) as pool:
        results = pool.map(process_func, sls_ls)
    
    # Separate valid data and collect valid infos only
    valid_data = []
    skipped_count = 0
    error_count = 0
    
    for stacked_img, infos, is_valid in results:
        if infos is None:  # Error case
            error_count += 1
            continue
        
        if is_valid:
            valid_data.append((stacked_img, infos))
        else:
            skipped_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  Total files: {total_files}")
    print(f"  Valid (GI_SNR >= {GI_SNR_threshold}): {len(valid_data)}")
    print(f"  Skipped (GI_SNR < {GI_SNR_threshold}): {skipped_count}")
    print(f"  Errors: {error_count}")
    
    # Save valid source info to CSV (only sources meeting SNR threshold)
    csv_path = os.path.join(output_dir, 'valid_sources.csv')
    if valid_data:
        valid_infos = [infos for _, infos in valid_data]
        keys = list(valid_infos[0].keys())
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(valid_infos)
        print(f"\nSaved info for {len(valid_infos)} valid sources to {csv_path}")
    
    # Calculate number of LMDB files needed for valid data
    num_valid = len(valid_data)
    if num_valid == 0:
        print("\nNo valid data to save to LMDB!")
    else:
        num_lmdb_files = (num_valid + data_per_file - 1) // data_per_file
        print(f"\nCreating {num_lmdb_files} LMDB file(s) for {num_valid} valid samples")
        
        # Write valid data to LMDB files (serialized, as LMDB requires)
        for lmdb_idx in range(num_lmdb_files):
            start_idx = lmdb_idx * data_per_file
            end_idx = min((lmdb_idx + 1) * data_per_file, num_valid)
            
            lmdb_path = os.path.join(output_dir, f'sls_specs_{lmdb_idx:04d}.lmdb')
            print(f"\nCreating {lmdb_path} with samples {start_idx} to {end_idx-1}")
            
            # Create LMDB environment with sufficient map_size
            # Each entry: stacked image (2x40x480x4 bytes) + metadata (~1KB) ≈ 155KB per sample
            # 10k samples ≈ 1.5GB, set map_size to 3GB to be safe
            map_size = 3 * 1024 * 1024 * 1024  # 3GB
            
            env = lmdb.open(lmdb_path, map_size=map_size)
            
            with env.begin(write=True) as txn:
                for idx in range(start_idx, end_idx):
                    stacked_img, infos = valid_data[idx]
                    
                    # Prepare data to store
                    data = {
                        'stacked_img': stacked_img.astype(np.float32),  # Shape: (2, 40, 480)
                        'infos': infos
                    }
                    
                    # Serialize data
                    data_bytes = pickle.dumps(data)
                    
                    # Store in LMDB with key as zero-padded index
                    key = f'{idx - start_idx:08d}'.encode('ascii')
                    txn.put(key, data_bytes)
                    
                    if ((idx - start_idx + 1) % 100 == 0):
                        print(f"  Written {idx - start_idx + 1}/{end_idx - start_idx} samples")
            
            env.close()
            print(f"Completed {lmdb_path}")
        
        print(f"\nAll done! Created {num_lmdb_files} LMDB file(s) in {output_dir}")
    
