import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Pattern

du_logger = logging.getLogger('main.data_utils')

# You may want to order by distribution of formatting if you have multiple date formats, as this is a bottleneck
__KNOWN_DATE_FORMATS = ['%m/%d/%Y %I:%M:%S %p']


def clean_col_names(df):
    '''Convert column names to lower case and replace spaces with underscores'''
    if pd.api.types.is_string_dtype(df.columns):
        df.columns = df.columns.str.lower().str.replace(' +', '_')


def remove_excess_white_space(df):
    '''
    Remove excess inner and outer whitespace from column values of DateFrame (inplace)
    
    Parameters:
    df (DataFrame): DataFrame to convert    
    '''
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = df[col].str.strip().str.replace(' +', ' ')


def parse_dt(dt, formats=__KNOWN_DATE_FORMATS):
    '''
    Try to parse a date using the supplied formats.
    
    Parameters:
    dt (str): string date
    
    Keyword arguments:
    formats (list of str): strptime style string formats (default -> local known formats)
    
    Returns:
    datetime if successfully parsed, else numpy nan    
    '''
    if pd.isna(dt): return np.nan
    
    for fmt in __KNOWN_DATE_FORMATS:
        try: dt = datetime.strptime(dt, fmt); break
        except ValueError: pass
    
    if isinstance(dt, datetime):
        return dt
    else:
        return np.nan
    
    
def process_field(df, field_name, rejects, 
                  other_nulls=[], pre_process=[], date_field=False, validation=None,  
                  post_process=[], valid_values=None, generated_cols=[], drop_field=False):
    '''
    Clean, validate and/or transform a DataFrame field (column) inplace.
    
    The process works as follows:
    1) Filter for non-null, non-empty string
    2) Filter for additional nulls in (optional) param other_nulls
    3) Apply any pre-processing on the field in (optional) param pre_processing
    4) Validate the field with (optional) param validation 
    5) Apply any post-processing (cosmetic changes) in (optional) param post_process
    6) Filter for valid values in (optional) param valid_values
    7) Generate any new columns from (optional) param generated_cols
    8) Update the supplied rejects dictionary with any problems
    9) Replace the field with the cleaned field, and create a new field with a reference to the original 
        (if validation failed (step 4) or valid values failed (step 6))
    10) (Optionally) drop the original field
    
    The resulting data frame will have clean values in the original field, and a new column of the form
    field_name + _orig, containing the original field value IF any validation error occured in step 4 or step 6
    
    Parameters:
    df (DataFrame): DataFrame to be manipulated (inplace)
    field_name: field in DataFrame to work on
    rejects (dict of <str, set of ints>): dict with history of errors to update on validation errors
    
    Keyword arguments:
    other_nulls (list of str): 
        additional values to be considered null
    pre_process (list of tuple(Regex Pattern, str)): 
        Each regex pattern will be search/replaced prior to validation
    date_field (bool): 
        If True validation will redirect to parse_dt
    validation (None or Regex Pattern or Lambda):
        If None, field is not validated (passed through)
        If Regex Pattern, a FULL match is considered valid
        If Lambda, lambda should take a series, and return a series of dtype bool
    post_process (list of tuple(Regex Pattern, str) or tuple(None, Lambda))
        If list of tuple(Regex Pattern, str), each pattern will be search/replaced
        If list of tuple(None, Lambda), the lambda should take a series and return a series
    valid_values (list of str)
        Anything not in the list of valid values will be rejected
    generated_cols (list of functions)
        Any supplied function should be of the form def some_function(df, series).  
        The current df will be passed along with the series, and any generated columns can be appended in your function
    drop_field (bool):
        If true, field will be dropped (e.g. if you're only interested in generated columns)
    '''
    field_orig = df.loc[df[field_name].notnull() & df[field_name].ne(''), field_name].copy()
    for other_null in other_nulls:
        field_orig = field_orig.loc[~field_orig.eq(other_null)]
        
    field = field_orig.copy()
    
    # If everything is null
    if len(field_orig) == 0:
        df[field_name] = np.nan
        df[field_name + '_orig'] = np.nan

        return

    # Pre Process (Clean data before validation)
    for pat, repl in pre_process:
        field = field.str.replace(pat, repl)
        
    # Validation
    if not date_field:
        if validation is not None and isinstance(validation, Pattern):
            field_mask = field.str.match(validation, na=False) # specifying na=false|true guaruntees the dtype will be bool
        elif validation is not None and callable(validation):
            field_mask = validation(field)
        else:
            field_mask = field.eq(field)
    else:
        field = field.apply(parse_dt)
        field_mask = field.notnull()
        
    # Update field to non-null and valid
    field = field.loc[field_mask]

    # Post Process (Cosmetic changes after validation)
    for pat, fn_or_str in post_process:
        if pat is not None and isinstance(pat, Pattern):
            field = field.str.replace(pat, fn_or_str)
        elif pat is None and callable(fn_or_str):
            field = fn_or_str(field)
        
    # Check valid values if present
    if valid_values is not None:
        field_mask = field_mask & field.isin(valid_values)
        field = field.loc[field_mask]
        
    # Apply custom functions to generate new fields 
    for fn in generated_cols:
        fn(df, field)

    # Update
    rejects[field_name].update(field_mask.loc[~field_mask].index)
    df[field_name + '_orig'] = field_orig.loc[~field_mask]
    if not drop_field:
        df[field_name] = field
    else:
        df.drop(field_name, axis='columns', inplace=True)

    
def analyze_hard_and_soft_rejects(hard_rejects, soft_rejects):
    '''
    Summarize rejected rows (hard rejects) and fields (soft rejects).
    
    Parameters:
    hard_rejects (dict of <str, list of int>): field names and list of hard reject indices
    soft_rejects (dict of <str, list of int>): field names and list of soft reject indices
    
    Returns:
    unique_hard_rejects (set of int): set intersection of all hard reject indices
    unique_soft_rejects (set of int): set intersection of all soft reject indices
    '''
    unique_hard_rejects = set()
    unique_soft_rejects = set()
    total_soft_fields_bad = 0

    for key in hard_rejects.keys():
        unique_hard_rejects = unique_hard_rejects | hard_rejects[key]
        du_logger.info(f'Hard rejects {key}: {len(hard_rejects[key]):,}')

    du_logger.info('')

    for key in soft_rejects.keys():
        unique_soft_rejects = unique_soft_rejects | soft_rejects[key]
        du_logger.info(f'Soft rejects {key}: {len(soft_rejects[key]):,}')
        total_soft_fields_bad += len(soft_rejects[key])

    du_logger.info('')
    du_logger.info(f'Hard Reject Total (entire row excluded): {len(unique_hard_rejects):,}')
    du_logger.info(f'Soft Reject Total (count of rows with bad data): {len(unique_soft_rejects):,}')
    du_logger.info(f'Bad Soft Fields (count of fields that will be nulled): {total_soft_fields_bad:,}')
    
    return unique_hard_rejects, unique_soft_rejects


def generate_reject_dfs(df, df_filt, file_name, hard_rejects, unique_hard_rejects, soft_rejects, unique_soft_rejects):
    '''
    Generate DFs for hard and soft rejects
    
    Parameters:
    df (DataFrame): the original unaltered DataFrame before any processing
    df_filt (DataFrame): the DataFrame filtered of hard rejects and processed for soft rejects
    hard_rejects (dict of col_name -> indices): column names and indices of offending rows for hard rejects
    unique_hard_rejects (set of indices): unique indices of any row resulting in a hard reject
    soft_rejects (dict of col_name -> indices): column names and indices of offending rows for soft rejects
    unique_soft_rejects (set of indices): unique indices of any row resulting in a soft reject
    
    Returns:
    hard_rej_df (DataFrame): filtered DataFrame with hard reject rows, along with the file name and columns that failed
    soft_rej_df (DataFrame): filtered DataFrame with soft reject rows, along with the file name and columns that failed
    '''
    hard_rej_df = df.loc[unique_hard_rejects].copy()
    hard_rej_df.insert(0, 'file_name', file_name)
    hard_rej_df.insert(1, 'cols', np.nan)
    cols = []
        
    for row_id in unique_hard_rejects:
        bad_cols = [key for key in hard_rejects.keys() if row_id in hard_rejects[key]]            
        cols.append(';'.join(bad_cols))
        
    hard_rej_df['cols'] = cols
    
    soft_rej_df = df_filt.loc[unique_soft_rejects].copy()
    soft_rej_df.insert(0, 'file_name', file_name)
    soft_rej_df.insert(1, 'cols', np.nan)
    cols.clear()
    
    for row_id in unique_soft_rejects:
        bad_cols = [key for key in soft_rejects.keys() if row_id in soft_rejects[key]]            
        cols.append(';'.join(bad_cols))
    
    soft_rej_df['cols'] = cols
                               
    return hard_rej_df, soft_rej_df
    

def split_file(file, has_headers=True, include_headers=True, headers=None, max_lines=1_000_000):
    '''
    Split a text file into smaller files
    
    Parameters:
    file (Pathlib Path to file): file to split up
    
    Keyword Arguments:
    has_headers (bool): True if file has headers
    include_headers (bool): True if you want to include headers with each split file
    headers (str): String with headers to use (either an override or for data without headers)
    max_lines (int): Max lines per file
    
    Returns:
    split_files (list of Pathlib Paths): list of Pathlib Paths of split files
    '''
    du_logger.info(f'Attempting to split file {file.name} into files with {max_lines:,} rows')
    du_logger.info(f'File has headers => {has_headers}, include headers => {include_headers}')
    if headers is not None:
        du_logger.info(f'Override headers: {headers}')
    
    split_files = []
    
    with file.open('r') as raw_in:
        if has_headers:
            if include_headers and headers is None:
                headers = raw_in.readline()
            
        line_cnt, out_file_cnt = 0, 0
        
        raw_out = None
        
        for line in raw_in:
            if line_cnt % max_lines == 0:
                line_cnt = 1 if include_headers and headers is not None else 0
                out_file_cnt += 1

                if raw_out is not None: raw_out.close()
                
                out_file = file.parent/file.name.replace(file.suffix, f'_{out_file_cnt:0>3d}{file.suffix}')
                du_logger.info(f'Writting to file {out_file.name}')
                split_files.append(out_file)
                raw_out = out_file.open('w')
                if include_headers and headers is not None: raw_out.write(f'{headers}')
            
            raw_out.write(line)            
            line_cnt += 1
    
    if raw_out is not None: raw_out.close()
    
    return split_files

