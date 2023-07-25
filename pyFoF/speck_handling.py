"""
Module for creating and handling speck files from the catalogs.
"""

import numpy as np
from astropy.table import Table


def convert_edge_data_to_speck() -> None:
    """Converts the output edge data file into a speck file."""
    pass

def create_group_speck(group_cat_fits: str, radii: np.ndarray = None) -> None:
    """
    Creates a speck file of the group catalog by creating wire mesh objects.
    """
    dat = Table.read(group_cat_fits)
    if radii is None:
        radii = np.ones(len(dat))*2

    with open(group_cat_fits.replace('.fits', '.speck'), 'w', encoding='utf8') as file:
        for i, row in enumerate(dat):
            file.write(
                f"{row['equi_x']} {row['equi_y']} {row['equi_z']} ellipsoid -r  {radii[i]} \
                      -c 10 -s wire -n 24 # {row['group_id']} \n")


def convert_table_to_particle_speck(fits_table: Table, outfile: str) -> None:
    """
    Writes a table into a particle speck file, which is specifically 
    useful for galaxy-type data.

    The table must have equi_x, equi_y, and equi_z values.
    """
    with open(outfile, 'w',  encoding='utf8') as file:
        # Header
        counter = 0
        additional_indicies = []
        for i, label in enumerate(list(fits_table.columns)):
            if label not in ['equi_x', 'equi_y', 'equi_z']:
                file.write(f'datavar {counter} {label} \n')
                counter += 1
                additional_indicies.append(i)

        # Body
        for row in fits_table:
            file.write(f'{row["equi_x"]} {row["equi_y"]} {row["equi_z"]} ')
            for idx in additional_indicies:
                file.write(f'{row[idx]} ')
            file.write(' \n')


def create_galaxy_speck(galaxy_cat_fits: str) -> None:
    """
    Creates two particle speck files, one for galaxies in groups and one for 
    galaxies not in groups.
    """
    dat = Table.read(galaxy_cat_fits)
    group_galaxy_ids = np.where(dat['group_id'] != -1)[0]
    field_galaxy_ids = np.where(dat['group_id'] == -1)[0]
    dat_group_galaxies = dat[group_galaxy_ids]
    dat_field_galaxies = dat[field_galaxy_ids]
    convert_table_to_particle_speck(
        dat_group_galaxies, galaxy_cat_fits.replace('.fits', '_group_gals.speck'))
    convert_table_to_particle_speck(
        dat_field_galaxies, galaxy_cat_fits.replace('.fits', '_field_gals.speck'))
