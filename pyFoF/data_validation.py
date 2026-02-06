"""Data validation module for astronomical data."""

import warnings
import pandas as pd
import numpy as np


def validate_ra(ra_values: pd.Series) -> bool:
    """
    Validate Right Ascension values.
    
    RA should be in the range [0, 360) degrees.
    
    Args:
        ra_values: Series containing RA values
        
    Returns:
        True if all values are valid, False otherwise
    """
    if ra_values.isna().all():
        return True
    
    valid_mask = (ra_values >= 0) & (ra_values < 360)
    invalid_count = (~valid_mask & ra_values.notna()).sum()
    
    if invalid_count > 0:
        warnings.warn(
            f"Found {invalid_count} RA values outside valid range [0, 360). "
            "RA values should be in degrees.",
            UserWarning
        )
        return False
    return True


def validate_dec(dec_values: pd.Series) -> bool:
    """
    Validate Declination values.
    
    Dec should be in the range [-90, 90] degrees.
    
    Args:
        dec_values: Series containing Dec values
        
    Returns:
        True if all values are valid, False otherwise
    """
    if dec_values.isna().all():
        return True
    
    valid_mask = (dec_values >= -90) & (dec_values <= 90)
    invalid_count = (~valid_mask & dec_values.notna()).sum()
    
    if invalid_count > 0:
        warnings.warn(
            f"Found {invalid_count} Dec values outside valid range [-90, 90]. "
            "Dec values should be in degrees.",
            UserWarning
        )
        return False
    return True


def validate_velocity(vel_values: pd.Series, allow_negative: bool = False) -> bool:
    """
    Validate velocity values.
    
    By default, velocity should be positive (recession velocity).
    
    Args:
        vel_values: Series containing velocity values
        allow_negative: If True, allows negative velocities (approaching)
        
    Returns:
        True if all values are valid, False otherwise
    """
    if vel_values.isna().all():
        return True
    
    if allow_negative:
        return True
    
    invalid_mask = vel_values < 0
    invalid_count = (invalid_mask & vel_values.notna()).sum()
    
    if invalid_count > 0:
        warnings.warn(
            f"Found {invalid_count} negative velocity values. "
            "Expected positive recession velocities.",
            UserWarning
        )
        return False
    return True


def validate_magnitude(
    mag_values: pd.Series, 
    min_mag: float = -5.0, 
    max_mag: float = 30.0
) -> bool:
    """
    Validate magnitude values.
    
    Args:
        mag_values: Series containing magnitude values
        min_mag: Minimum expected magnitude (brightest)
        max_mag: Maximum expected magnitude (faintest)
        
    Returns:
        True if all values are valid, False otherwise
    """
    if mag_values.isna().all():
        return True
    
    valid_mask = (mag_values >= min_mag) & (mag_values <= max_mag)
    invalid_count = (~valid_mask & mag_values.notna()).sum()
    
    if invalid_count > 0:
        warnings.warn(
            f"Found {invalid_count} magnitude values outside expected range "
            f"[{min_mag}, {max_mag}].",
            UserWarning
        )
        return False
    return True


def validate_astronomical_data(
    df: pd.DataFrame,
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    vel_col: str = None,
    mag_col: str = None,
    mag_min: float = -5.0,
    mag_max: float = 30.0,
    allow_negative_vel: bool = False,
) -> dict:
    """
    Validate astronomical data in a DataFrame.
    
    Args:
        df: DataFrame containing astronomical data
        ra_col: Name of the RA column
        dec_col: Name of the Dec column
        vel_col: Name of the velocity column (optional)
        mag_col: Name of the magnitude column (optional)
        mag_min: Minimum expected magnitude
        mag_max: Maximum expected magnitude
        allow_negative_vel: Whether to allow negative velocities
        
    Returns:
        Dictionary with validation results for each field
    """
    results = {}
    
    # Try to auto-detect columns if not specified
    if ra_col not in df.columns:
        ra_candidates = [c for c in df.columns if c.lower() in ['ra', 'right_ascension', 'rightascension']]
        if ra_candidates:
            ra_col = ra_candidates[0]
    
    if dec_col not in df.columns:
        dec_candidates = [c for c in df.columns if c.lower() in ['dec', 'de', 'declination']]
        if dec_candidates:
            dec_col = dec_candidates[0]
    
    # Validate RA
    if ra_col in df.columns:
        try:
            ra_series = pd.to_numeric(df[ra_col], errors='coerce')
            results['ra'] = validate_ra(ra_series)
        except Exception as e:
            warnings.warn(f"Could not validate RA column '{ra_col}': {e}", UserWarning)
            results['ra'] = False
    
    # Validate Dec
    if dec_col in df.columns:
        try:
            dec_series = pd.to_numeric(df[dec_col], errors='coerce')
            results['dec'] = validate_dec(dec_series)
        except Exception as e:
            warnings.warn(f"Could not validate Dec column '{dec_col}': {e}", UserWarning)
            results['dec'] = False
    
    # Validate velocity
    if vel_col and vel_col in df.columns:
        try:
            vel_series = pd.to_numeric(df[vel_col], errors='coerce')
            results['velocity'] = validate_velocity(vel_series, allow_negative_vel)
        except Exception as e:
            warnings.warn(f"Could not validate velocity column '{vel_col}': {e}", UserWarning)
            results['velocity'] = False
    
    # Validate magnitude
    if mag_col and mag_col in df.columns:
        try:
            mag_series = pd.to_numeric(df[mag_col], errors='coerce')
            results['magnitude'] = validate_magnitude(mag_series, mag_min, mag_max)
        except Exception as e:
            warnings.warn(f"Could not validate magnitude column '{mag_col}': {e}", UserWarning)
            results['magnitude'] = False
    
    return results
