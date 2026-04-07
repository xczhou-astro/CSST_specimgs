#!/usr/bin/env python3
"""
High-performance coefficient extraction from FITS files
CPU-optimized version with memory mapping and numba acceleration
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from numba import jit, prange
import math
from tqdm import tqdm

@jit(nopython=True, parallel=True)
def find_matches_parallel(df_ids, primary_targetid, output_indices):
    """Find matches between df_ids and primary_targetid using numba parallelization"""
    for i in prange(len(df_ids)):
        target_id = df_ids[i]
        # Default is -1 (not found)
        match_idx = -1
        
        # Binary search would be ideal, but requires sorted data
        # Using linear search for compatibility
        for j in range(len(primary_targetid)):
            if primary_targetid[j] == target_id:
                match_idx = j
                break
                
        output_indices[i] = match_idx

def ensure_native_byteorder(array):
    """Convert array to native byte order if needed"""
    if array.dtype.byteorder not in ('=', '|'):
        return array.astype(array.dtype.newbyteorder('='))
    return array

def main():
    start_time = time.time()
    print("Starting optimized coefficient extraction...")
    
    # Path to data files
    fits_path = 'DESI/zall-pix-iron.fits'
    # csv_path = 'desi_spring_matched_target_inf.csv'
    csv_path = 'matched_sources/desi_spring_matched_selected.csv'
    
    # Prefer memory mapping for large FITS files to reduce memory usage
    print(f"Memory-mapping FITS file: {fits_path}")
    with fits.open(fits_path, memmap=True) as hdul:
        # Extract only necessary data to minimize memory usage
        print("Extracting targetid and primary flag data...")
        desi_targetid = hdul[1].data['targetid']
        desi_primary = hdul[1].data['zcat_primary']
        
        # Filter by primary flag and save original indices
        primary_indices = np.where(desi_primary)[0]
        primary_targetid = desi_targetid[primary_indices]
        
        # Load CSV data
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df_ids = df['desi_id'].values
        
        # Prepare output array for match indices
        match_indices = np.full(len(df_ids), -1, dtype=np.int32)
        
        # Fix data types for Numba compatibility
        print("Converting data to Numba-compatible format...")
        primary_targetid_native = ensure_native_byteorder(primary_targetid)
        df_ids_native = ensure_native_byteorder(df_ids)
        
        # Print data type information for debugging
        print(f"Original primary_targetid dtype: {primary_targetid.dtype}")
        print(f"Converted primary_targetid dtype: {primary_targetid_native.dtype}")
        print(f"Original df_ids dtype: {df_ids.dtype}")
        print(f"Converted df_ids dtype: {df_ids_native.dtype}")
        
        # Convert to explicit types that Numba supports well
        primary_targetid_native = primary_targetid_native.astype(np.int64)
        df_ids_native = df_ids_native.astype(np.int64)
        
        # Use numba to accelerate matching
        print("Finding matches using parallel processing...")
        try:
            find_matches_parallel(df_ids_native, primary_targetid_native, match_indices)
        except Exception as e:
            print(f"Numba acceleration failed: {e}")
            print("Falling back to non-accelerated matching...")
            
            # Non-accelerated fallback
            for i in tqdm(range(len(df_ids_native)), desc="Finding matches"):
                target_id = df_ids_native[i]
                for j in range(len(primary_targetid_native)):
                    if primary_targetid_native[j] == target_id:
                        match_indices[i] = j
                        break
        
        # Count matches for reporting
        match_count = np.sum(match_indices >= 0)
        print(f"Found {match_count} matches out of {len(df_ids)} IDs ({match_count/len(df_ids)*100:.2f}%)")
        
        # Process in efficient batches to minimize memory impact
        print("Extracting coefficients and shape_r in batches...")
        batch_size = 1000  # Adjust based on available memory
        num_batches = math.ceil(len(df_ids) / batch_size)
        
        # Preallocate results list for better performance
        coeffs = [""] * len(df_ids)
        shape_r = np.full(len(df_ids), np.nan, dtype=np.float32)
        
        # Process batches with progress bar
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(df_ids))
            batch_indices = match_indices[batch_start:batch_end]
            
            # Find the indices that matched (not -1)
            valid_match_mask = batch_indices >= 0
            valid_batch_positions = np.where(valid_match_mask)[0]
            
            # Convert matched batch indices to original FITS file indices
            valid_primary_indices = []
            for pos in valid_batch_positions:
                match_idx = batch_indices[pos]
                if match_idx >= 0 and match_idx < len(primary_indices):
                    valid_primary_indices.append(primary_indices[match_idx])
            
            # If there are matches in this batch
            if len(valid_primary_indices) > 0:
                try:
                    # Read coefficients and shape_r only for matches, minimizing I/O
                    coeff_data = hdul[1].data['coeff'][valid_primary_indices]
                    shape_r_data = hdul[1].data['shape_r'][valid_primary_indices]
                    
                    # Process each matched item
                    for i, batch_pos in enumerate(valid_batch_positions):
                        if i < len(coeff_data):  # Safety check
                            global_pos = batch_start + batch_pos
                            coeff = coeff_data[i]
                            # Faster string joining
                            coeff_str = ','.join(str(x) for x in coeff)
                            coeffs[global_pos] = coeff_str
                            shape_r[global_pos] = shape_r_data[i]
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Leave these as empty strings/NaN
    
    # Add coefficients and shape_r to dataframe
    df['coeff'] = coeffs
    # df['shape_r'] = shape_r
    
    # Save results
    # output_file = 'matched_sources/desi_spring_lt_1arcsec_coeff.csv'
    output_file = 'matched_sources/desi_spring_matched_selected_coeff.csv'
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
