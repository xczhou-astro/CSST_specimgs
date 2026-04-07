import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const
from astropy.table import Table
import os
import argparse

desi = fits.open('DESI/zall-pix-iron.fits')

idx = (desi[1].data['zcat_primary']) & (desi[1].data['zwarn'] == 0) \
    & (desi[1].data['spectype'] == 'GALAXY') & (desi[1].data['z'] > 0.01) \
    & (desi[1].data['maskbits'] == 0) & (desi[1].data['flux_g'] > 0) \
    & (desi[1].data['flux_r'] > 0) & (desi[1].data['flux_z'] > 0) \
    & (desi[1].data['flux_ivar_g'] > 0) & (desi[1].data['flux_ivar_r'] > 0) \
    & (desi[1].data['flux_ivar_z'] > 0) & (desi[1].data['fiberflux_g'] > 0) \
    & (desi[1].data['fiberflux_r'] > 0) & (desi[1].data['fiberflux_z'] > 0) \

desi_id = desi[1].data['targetid'][idx]
desi_ra = desi[1].data['target_ra'][idx]
desi_dec = desi[1].data['target_dec'][idx]
desi_z = desi[1].data['z'][idx]
desi_flux_g = desi[1].data['flux_g'][idx]
desi_flux_r = desi[1].data['flux_r'][idx]
desi_flux_z = desi[1].data['flux_z'][idx]
desi_fiberflux_g = desi[1].data['fiberflux_g'][idx]
desi_fiberflux_r = desi[1].data['fiberflux_r'][idx]
desi_fiberflux_z = desi[1].data['fiberflux_z'][idx]
desi_shape_r = desi[1].data['shape_r'][idx]

desi_flux_g = 22.5 - 2.5 * np.log10(desi_flux_g) # convert to AB magnitudes
desi_flux_r = 22.5 - 2.5 * np.log10(desi_flux_r)
desi_flux_z = 22.5 - 2.5 * np.log10(desi_flux_z)
desi_fiberflux_g = 22.5 - 2.5 * np.log10(desi_fiberflux_g)
desi_fiberflux_r = 22.5 - 2.5 * np.log10(desi_fiberflux_r)
desi_fiberflux_z = 22.5 - 2.5 * np.log10(desi_fiberflux_z)

desi_coords = SkyCoord(ra=desi_ra*u.degree, dec=desi_dec*u.degree, frame='icrs')

hsc = pd.read_csv('HSC_catalog/HSC_spring.csv.gz')
hsc = hsc.rename(columns={'# object_id': 'object_id'})

hsc_id = hsc['object_id'].values
hsc_ra = hsc['ra'].values
hsc_dec = hsc['dec'].values
i_flux = hsc['i_cmodel_flux'].values # in nano jansky
i_flux_err = hsc['i_cmodel_fluxerr'].values
i_mag = hsc['i_cmodel_mag'].values

idx = np.where((i_flux > 0) & (i_flux_err > 0) & (i_flux / i_flux_err > 5)
               & (i_mag < 24.0))[0]

hsc_ra = hsc_ra[idx]
hsc_dec = hsc_dec[idx]
hsc_id = hsc_id[idx]
g_flux = hsc['g_cmodel_flux'].values[idx]
g_flux_err = hsc['g_cmodel_fluxerr'].values[idx]
r_flux = hsc['r_cmodel_flux'].values[idx]
r_flux_err = hsc['r_cmodel_fluxerr'].values[idx]
i_flux = i_flux[idx]
i_flux_err = i_flux_err[idx]
i_mag = i_mag[idx]
z_flux = hsc['z_cmodel_flux'].values[idx]
z_flux_err = hsc['z_cmodel_fluxerr'].values[idx]

hsc_coords = SkyCoord(ra=hsc_ra*u.degree, dec=hsc_dec*u.degree, frame='icrs')

match_idx, d2d, _ = hsc_coords.match_to_catalog_sky(desi_coords)
idx_d2d = d2d < 1.5 * u.arcsec
match_idx = match_idx[idx_d2d]

print(np.sum(idx_d2d))

desi_ra = desi_ra[match_idx]
desi_dec = desi_dec[match_idx]
desi_z = desi_z[match_idx]
desi_id = desi_id[match_idx]
desi_flux_g = desi_flux_g[match_idx]
desi_flux_r = desi_flux_r[match_idx]
desi_flux_z = desi_flux_z[match_idx]
desi_fiberflux_g = desi_fiberflux_g[match_idx]
desi_fiberflux_r = desi_fiberflux_r[match_idx]
desi_fiberflux_z = desi_fiberflux_z[match_idx]
desi_shape_r = desi_shape_r[match_idx]

hsc_ra = hsc_ra[idx_d2d]
hsc_dec = hsc_dec[idx_d2d]
hsc_id = hsc_id[idx_d2d]
hsc_g_flux = g_flux[idx_d2d]
hsc_g_flux_err = g_flux_err[idx_d2d]
hsc_g_mag = -2.5 * np.log10(hsc_g_flux * 10**-9) + 8.90 # in AB
hsc_r_flux = r_flux[idx_d2d]
hsc_r_flux_err = r_flux_err[idx_d2d]
hsc_r_mag = -2.5 * np.log10(hsc_r_flux * 10**-9) + 8.90 # in AB
hsc_i_flux = i_flux[idx_d2d]
hsc_i_flux_err = i_flux_err[idx_d2d]
hsc_i_mag = i_mag[idx_d2d]
hsc_z_flux = z_flux[idx_d2d]
hsc_z_flux_err = z_flux_err[idx_d2d]
hsc_z_mag = -2.5 * np.log10(hsc_z_flux * 10**-9) + 8.90 # in AB

print(desi_z.shape)
print(hsc_i_mag.shape)

df = pd.DataFrame({
    'desi_ra': desi_ra,
    'desi_dec': desi_dec,
    'desi_z': desi_z,
    'desi_id': desi_id,
    'desi_flux_g': desi_flux_g,
    'desi_flux_r': desi_flux_r,
    'desi_flux_z': desi_flux_z,
    'desi_fiberflux_g': desi_fiberflux_g,
    'desi_fiberflux_r': desi_fiberflux_r,
    'desi_fiberflux_z': desi_fiberflux_z,
    'desi_shape_r': desi_shape_r,
    'hsc_ra': hsc_ra,
    'hsc_dec': hsc_dec,
    'hsc_id': hsc_id,
    'hsc_g_mag': hsc_g_mag,
    'hsc_g_flux': hsc_g_flux,
    'hsc_g_flux_err': hsc_g_flux_err,
    'hsc_r_mag': hsc_r_mag,
    'hsc_r_flux': hsc_r_flux,
    'hsc_r_flux_err': hsc_r_flux_err,
    'hsc_i_mag': hsc_i_mag,
    'hsc_i_flux': hsc_i_flux,
    'hsc_i_flux_err': hsc_i_flux_err,
    'hsc_z_mag': hsc_z_mag,
    'hsc_z_flux': hsc_z_flux,
    'hsc_z_flux_err': hsc_z_flux_err
})


df.to_csv('matched_sources/desi_spring_griz_i_mag_24.csv', index=False)
