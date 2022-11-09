"""Data class to handle inputs and outputs."""

from astropy.table import Table
import pandas as pd

def read_in_fits_table(fits_table_name: str) -> pd.DataFrame:
    """reads in a fits table."""
    fits_table = Table.read(fits_table_name)
    pandas_df = fits_table.to_pandas()
    return pandas_df



def _read_header_from_iaupac_table(readlines_object):
    """Finds the header and reads in those values as the variables"""
    for line in readlines_object:
        if line[0] == '|':
            header_data = [x.strip() for x in line.split('|') if len(x.strip()) != 0]
            return header_data
    return 'No Header Found!'

def _find_data_of_iaupac_data(readlines_object):
    """Finds the first index which isn't part of the header"""
    for i, line in enumerate(readlines_object):
        if line[0] not in ['#', '/', '|', '\\']:
            return readlines_object[i:]
    return 'No Data Found!'

def read_in_iaupac_table(iaupac_name: str):
    """Reads in an IAUPAC table"""
    with open(iaupac_name, encoding = 'utf-8') as file:
        text = file.readlines()
    header = _read_header_from_iaupac_table(text)
    data = _find_data_of_iaupac_data(text)
    return header, data





def check_file_type(file_name: str) -> str:
    """Identifies the file extension and runs the correct reader."""
    extension = file_name.split('.')[-1]


if __name__ == '__main__':
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'
    df = read_in_fits_table(INFILE)
