from astropy.io import fits, ascii
import pandas as pd
import numpy as np
from astropy.table import Table

# Create a pandas dataframe from the provided data
data = {
    'idname': [94967, 94976, 94982], 
    'ra': [3.62163, 3.54946, 3.53893], 
    'dec': [-33.11417, -32.39796, -32.71177],
    'zcmb': [0.03029, 0.05788, 0.06229],
    'zhelio': [0.0312, 0.0588, 0.0632],
    'W1': [15.275, 14.542, 16.445],
    'eW1': [0.056, 0.034, 0.060]
}
df = pd.DataFrame(data)

# Convert the pandas dataframe to an astropy Table
table = Table.from_pandas(df)

# Save to FITS file
fits_filename = './data/Test_Data/Test_Data_Handling.fits'
table.write(fits_filename, format='fits', overwrite=True)

# Save to IAUPAC (ASCII) file
ascii_filename = './data/Test_Data/Test_Data_Handling.tbl'
table.write(ascii_filename, format='ascii.ipac', overwrite=True)
