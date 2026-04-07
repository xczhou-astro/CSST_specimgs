import pandas as pd
import numpy as np
from astropy.io import fits
import os

os.makedirs('output', exist_ok=True)

valid_catalog = pd.read_csv('output/lmdb_data/valid_sources.csv')
print(valid_catalog.shape)

desi = fits.open('../datasets/DESI/zall-pix-iron.fits')

# Build DESI lookup: targetid -> flux_w1, flux_w2 (convert to native byte order for pandas)
desi_data = desi[1].data

idx = (desi[1].data['zcat_primary']) & (desi[1].data['zwarn'] == 0) \
    & (desi[1].data['spectype'] == 'GALAXY') & (desi[1].data['z'] > 0.01) \
    & (desi[1].data['maskbits'] == 0) & (desi[1].data['flux_g'] > 0) \
    & (desi[1].data['flux_r'] > 0) & (desi[1].data['flux_z'] > 0) \
    & (desi[1].data['flux_ivar_g'] > 0) & (desi[1].data['flux_ivar_r'] > 0) \
    & (desi[1].data['flux_ivar_z'] > 0) & (desi[1].data['fiberflux_g'] > 0) \
    & (desi[1].data['fiberflux_r'] > 0) & (desi[1].data['fiberflux_z'] > 0)

targetid = np.asarray(desi_data['targetid'], dtype=np.int64)[idx]
flux_w1 = np.asarray(desi_data['flux_w1'], dtype=np.float64)[idx]
flux_w2 = np.asarray(desi_data['flux_w2'], dtype=np.float64)[idx]
flux_g = np.asarray(desi_data['flux_g'], dtype=np.float64)[idx]
flux_r = np.asarray(desi_data['flux_r'], dtype=np.float64)[idx]
flux_z = np.asarray(desi_data['flux_z'], dtype=np.float64)[idx]
fiberflux_g = np.asarray(desi_data['fiberflux_g'], dtype=np.float64)[idx]
fiberflux_r = np.asarray(desi_data['fiberflux_r'], dtype=np.float64)[idx]
fiberflux_z = np.asarray(desi_data['fiberflux_z'], dtype=np.float64)[idx]

# Magnitudes: use NaN for zero/negative flux to avoid log10 warnings
with np.errstate(divide='ignore', invalid='ignore'):
    desi_ra = np.asarray(desi_data['target_ra'], dtype=np.float64)[idx]
    desi_dec = np.asarray(desi_data['target_dec'], dtype=np.float64)[idx]
    desi_w1_mag = np.where(flux_w1 > 0, 22.5 - 2.5 * np.log10(flux_w1), np.nan)
    desi_w2_mag = np.where(flux_w2 > 0, 22.5 - 2.5 * np.log10(flux_w2), np.nan)
    desi_g_mag = np.where(flux_g > 0, 22.5 - 2.5 * np.log10(flux_g), np.nan)
    desi_r_mag = np.where(flux_r > 0, 22.5 - 2.5 * np.log10(flux_r), np.nan)
    desi_z_mag = np.where(flux_z > 0, 22.5 - 2.5 * np.log10(flux_z), np.nan)
    desi_g_fibermag = np.where(fiberflux_g > 0, 22.5 - 2.5 * np.log10(fiberflux_g), np.nan)
    desi_r_fibermag = np.where(fiberflux_r > 0, 22.5 - 2.5 * np.log10(fiberflux_r), np.nan)
    desi_z_fibermag = np.where(fiberflux_z > 0, 22.5 - 2.5 * np.log10(fiberflux_z), np.nan)
    desi_shape_r = np.asarray(desi_data['shape_r'], dtype=np.float64)[idx]

desi_df = pd.DataFrame({
    'desi_ra': desi_ra,
    'desi_dec': desi_dec,
    'targetid': targetid,
    'desi_g_mag': desi_g_mag,
    'desi_r_mag': desi_r_mag,
    'desi_z_mag': desi_z_mag,
    'desi_w1_mag': desi_w1_mag,
    'desi_w2_mag': desi_w2_mag,
    'desi_g_fibermag': desi_g_fibermag,
    'desi_r_fibermag': desi_r_fibermag,
    'desi_z_fibermag': desi_z_fibermag,
    'desi_shape_r': desi_shape_r,
})
desi.close()

# Merge on desi_id (sls) = targetid (desi), keep all sls rows
valid_catalog = valid_catalog.merge(
    desi_df,
    left_on='ID',
    right_on='targetid',
    how='left',
).drop(columns=['targetid'], errors='ignore')

valid_catalog.to_csv('output/lmdb_data/valid_sources_complete.csv', index=False)
print(valid_catalog.shape)