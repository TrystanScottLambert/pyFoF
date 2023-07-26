"""
Module for creating and handling speck files from the catalogs.
"""

import numpy as np
from astropy.table import Table


def _write_mesh_speck(
        position_ones: list[list[float, float, float]],
        positon_twos: list[list[float, float, float]],
        color: float, outfile: str, header: str) -> None:
    """Takes two lists of positions with x, y, z data and created a mesh speck file."""
    with open(outfile, 'w', encoding='utf8') as file:
        file.write(header)
        for pos1, pos2 in zip(position_ones, positon_twos):
            file.write(f'mesh -c {color} -s wire' + ' { \n 1 2 \n')
            for coord in pos1:
                file.write(f'{coord} ')
            file.write(' \n')
            for coord in pos2:
                file.write(f'{coord} ')
            file.write(' \n } \n\n')

def convert_edge_data_to_speck(galaxy_cat_fits: str, edge_data_file: str, number_bins: int) -> None:
    """Converts the output edge data file into a speck file."""
    galaxy_table = Table.read(galaxy_cat_fits)
    id_1, id_2, weights = np.loadtxt(edge_data_file, unpack=True)
    bins = np.arange(0, 1 + 1./number_bins, 1./number_bins)
    colors = np.linspace(1, 25, number_bins)
    for i in range(len(bins) -1):
        position_1 = []
        position_2 = []
        cut = np.where((weights > bins[i]) & (weights <= bins[i+1]))[0]
        pos_id_1 = id_1[cut]
        pos_id_2 = id_2[cut]
        for pos_1, pos_2 in zip(pos_id_1, pos_id_2):
            position_1.append([galaxy_table[int(pos_1)]['equi_x'],
                               galaxy_table[int(pos_1)]['equi_y'],
                               galaxy_table[int(pos_1)]['equi_z']])

            position_2.append([galaxy_table[int(pos_2)]['equi_x'],
                               galaxy_table[int(pos_2)]['equi_y'],
                               galaxy_table[int(pos_2)]['equi_z']])

        _write_mesh_speck(
            position_1, position_2, colors[i],
            f'edge_{round(bins[i+1], 2)}',
            f'#Edges ({bins[i]} {bins[i+1]}] \n\n')


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
