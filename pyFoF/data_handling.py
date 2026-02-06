"""Data class to handle inputs and outputs."""

from astropy.table import Table
import pandas as pd
from typing import Union
import ast
import numpy as np

from pyFoF.data_validation import validate_astronomical_data

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
            text_data = readlines_object[i:]
            data = [row.split() for row in text_data]
            return data
    return 'No Data Found!'

def read_in_iaupac_table(iaupac_name: str):
    """Reads in an IAUPAC table"""
    with open(iaupac_name, encoding = 'utf-8') as file:
        text = file.readlines()
    header = _read_header_from_iaupac_table(text)
    data = _find_data_of_iaupac_data(text)
    d_f = pd.DataFrame(data, columns = header)
    return d_f

EXTENSIONS = {
    'fits': read_in_fits_table,
    'tbl': read_in_iaupac_table,
}

def check_file_type(file_name: str) -> str:
    """Identifies the file extension and runs the correct reader."""
    extension = file_name.split('.')[-1]
    return extension


# def auto_convert_df_types(data_frame: pd.DataFrame) -> pd.DataFrame:
#     """Automatically assigns types to a data frame."""
#     for col in data_frame.columns:
#         if isinstance(list(data_frame[col])[0], int):
#             print('hey1')
#             data_frame[col] = data_frame[col].astype(int)

#         elif isinstance(list(data_frame[col])[0], float):
#             print('hey2')
#             data_frame[col] = data_frame[col].astype(float)

#         elif isinstance(list(data_frame[col])[0], str):
#             print('hey3')
#             data_frame[col] = data_frame[col].astype(str)
        
#         elif isinstance(list(data_frame[col])[0], bytes):
#             print('hey4')
#             data_frame[col] = data_frame[col].str.decode('utf-8')

#     return data_frame

def infer_dtype(value: str) -> Union[np.dtype, type]:
    if not value.strip():
        return np.dtype('O')
    
    try:
        val = type(ast.literal_eval(value))
        if val is int:
            return np.dtype('int64')
        elif val is float:
            return np.dtype('float64')
    except (SyntaxError, ValueError):
        pass

    return np.dtype('O')

def auto_convert_df_types(data_frame: pd.DataFrame) -> pd.DataFrame:
    for col in data_frame.columns:
        inferred_dtype = None

        if data_frame[col].dtype != str and data_frame[col].dtype != 'O':
            continue

        first_value = list(data_frame[col])[0]
        dtype = infer_dtype(first_value)

        if dtype == str:
            inferred_dtype = None
        else:
            inferred_dtype = dtype

        if inferred_dtype:
            try:
                data_frame[col] = data_frame[col].astype(inferred_dtype)
            except:
                data_frame[col] = data_frame[col].astype(str)

    return data_frame

def read_data(file_name: str, validate: bool = True) -> pd.DataFrame:
    """Reads in data of any type and returns a data frame.
    
    Args:
        file_name: Path to the data file
        validate: If True, performs astronomical data validation and issues
                 warnings for invalid values
                 
    Returns:
        DataFrame containing the read data
    """
    ext = check_file_type(file_name)
    d_f = EXTENSIONS[ext](file_name)
    d_f = auto_convert_df_types(d_f)
    
    if validate:
        validate_astronomical_data(d_f)
    
    return d_f

if __name__ == '__main__':
    INFILE = './data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    INFILE_FITS = './data/Kids/WISE-SGP_redshifts_w1mags.fits'

    df = read_data(INFILE)
    print(df.dtypes)
    print(df.head(3))
    df_fits = read_data(INFILE_FITS)
    print(df_fits.dtypes)
    print(df_fits.head(3))
    
