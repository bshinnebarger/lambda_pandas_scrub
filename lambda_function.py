import logging
import re
import os
import sys

from pathlib import Path
from collections import defaultdict, namedtuple, Counter
from datetime import datetime

import pandas as pd
import numpy as np
import s3fs

import aws_utils
import data_utils


ScrubRE = namedtuple('ScrubRE', ['TWO_LETTERS',
                                 'BLOCK', 'IUCR', 'PRIMARY_TYPE', 'DESCRIPTION', 'LOCATION_DESCRIPTION', 
                                 'LOCATION', 'ZIP_CODES'])

# Generic and column specific regex patterns for validation and transformation purposes
MY_REGX = ScrubRE(TWO_LETTERS=re.compile(r'^[a-z]{2}$', flags=re.I),                      
                      # Look for some kind of address like 013XX and then a street location like W 3RD AVE
                      # Will use to validate and to extract
                      BLOCK=re.compile(r'^(\d{1,4}X{1,4}) ((?:[a-z\d] ?){1,20}){1,5}$', flags=re.I),
                      # IUCR we just want to confirm it's some 4 length alphanumeric code
                      IUCR=re.compile(r'^[a-z\d]{4}$', flags=re.I),
                      # Look for up to five groups of letters, dashes
                      PRIMARY_TYPE=re.compile(r'^(?:[a-z\-]{1,20}(?: |$)){1,5}$', flags=re.I),
                      # Look for up to five groups of letters, numbers, or [-/:,()$]
                      DESCRIPTION=re.compile(r'^(?:[a-z\-\/\:\,\.\(\)\d\$}]{1,25}(?: |$)){1,7}$', flags=re.I),
                      # Look for up to five groups of letters, or [-/.,()]
                      LOCATION_DESCRIPTION=re.compile(r'^(?:[a-z\-\/\.\,\(\)]{1,20}(?: |$)){1,7}$', flags=re.I),
                      LOCATION=re.compile(r'^\((-?\d+\.\d+), ?(-?\d+\.\d+)\)$'),
                      # Zip codes should be length 4 or 5
                      ZIP_CODES=re.compile(r'^\d{5}|\d{4}$')
                      )

logger = logging.getLogger()
logger.name = 'main'
logger.setLevel(logging.INFO)

for handler in logger.handlers:
    if '%(aws_request_id)s' in handler.formatter._fmt:
        new_fmt = handler.formatter._fmt.replace('%(aws_request_id)s', '%(aws_request_id)s  %(name)s  %(funcName)s')
        handler.setFormatter(logging.Formatter(new_fmt))


def lambda_handler(event, context):
    
    s3_bucket = event['s3_bucket']
    s3_key = event['s3_key']
    s3_bucket_key = f'{s3_bucket}/{s3_key}'
    logger.info(f'Processing {s3_bucket_key}')
    
    s3 = s3fs.S3FileSystem()
  
    with s3.open(s3_bucket_key, 'r') as file:
        df = pd.read_csv(file, low_memory=False, encoding='utf-8', dtype=str)
    
    data_utils.clean_col_names(df)
    
    keep_cols = ['id', 'case_number', 'date', 'block', 'iucr', 'primary_type', 'description', 'location_description',
                'arrest', 'domestic', 'beat', 'district', 'ward', 'community_area', 'location', 'zip_codes']
    
    df = df[keep_cols]
    
    data_utils.remove_excess_white_space(df)
    
    df_filt, hard_rejects = process_hard_rejects(df)
    soft_rejects = process_soft_rejects(df_filt)
    unq_hard_rejects, unq_soft_rejects = data_utils.analyze_hard_and_soft_rejects(hard_rejects, soft_rejects)
    
    file_name = s3_bucket_key.split('/')[-1]
    hard_rej_df, soft_rej_df = data_utils.generate_reject_dfs(df, df_filt, file_name, hard_rejects, unq_hard_rejects, soft_rejects, unq_soft_rejects)
    
    s3_hard_up_bucket_key = f'{s3_bucket}/rejects/hard_rejects_{file_name}'
    s3_soft_up_bucket_key = f'{s3_bucket}/rejects/soft_rejects_{file_name}'
    s3_clean_up_bucket_key = f'{s3_bucket}/clean_data/clean_{file_name}'
    
    logger.info(f'Uploading hard rejects to {s3_hard_up_bucket_key}')
    logger.info(f'Uploading soft rejects to {s3_soft_up_bucket_key}')
    logger.info(f'Uploading clean data to {s3_clean_up_bucket_key}')
    
    with s3.open(s3_hard_up_bucket_key, 'w') as hard_up, \
            s3.open(s3_soft_up_bucket_key, 'w') as soft_up, \
            s3.open(s3_clean_up_bucket_key, 'w') as clean_up:
        hard_rej_df.to_csv(hard_up, index_label='file_index', encoding='utf-8')
        soft_rej_df.to_csv(soft_up, index_label='file_index', encoding='utf-8')
        
        clean_cols = [col for col in df_filt.columns if '_orig' not in col]
        df_filt[clean_cols].to_csv(clean_up, index=False, encoding='utf-8')
    
    logger.info('Success')
    
    return 'success'
  

def process_hard_rejects(df):
    
    df_filt = df.copy()
    
    logger.info(f'Length of records before hard rejects {len(df_filt):,}')
    
    hard_rejects = defaultdict(set)

    # Column specific logic
    # id -> Not null, must be digits
    id_val = lambda series: series.str.isdigit()
    # case_number -> Not null, first two must be alpha
    case_number_val = lambda series: series.str.slice(0,2).str.match(MY_REGX.TWO_LETTERS)
    
    # date -> Not null, try to parse known formats, also want to extract year, month
    def date_yr_mo(df, dates):
        df['year'] = dates.dt.year.astype(str)
        df['month'] = dates.dt.month.astype(str)

    nonnull_fields = {}
    nonnull_fields['id'] = {'validation': id_val}
    nonnull_fields['case_number'] = {'validation': case_number_val}
    nonnull_fields['date'] = {'date_field': True, 'other_nulls': ['0000-00-00'], 'generated_cols': [date_yr_mo]}

    for field, params in nonnull_fields.items():
        logger.debug(f'Processing {field}')
        data_utils.process_field(df_filt, field, hard_rejects, **params)

    # Filter out hard rejects
    for _, ids in hard_rejects.items():
        if len(ids) > 0: 
            df_filt = df_filt.loc[~df_filt.index.isin(ids)]

    logger.info(f'Length of records after hard rejects {len(df_filt):,}')
    
    return df_filt, hard_rejects
    
    
def process_soft_rejects(df_filt):

    soft_rejects = defaultdict(set)

    # Generic cosmetic changes (post-processing)
    capitalize_first = lambda series: series.str.title()
    upper_case = lambda series: series.str.upper()

    # Block we want to extract the hidden house number from the street location so we can analyze crime by street
    def block_num_addr(df, blocks):
        df[['house_num', 'street_addr']] = blocks.str.extract(MY_REGX.BLOCK, expand=True)

    primary_type_post = [(None, capitalize_first)]
    # Combine regex validation and max length 50
    description_val = lambda series: series.str.match(MY_REGX.DESCRIPTION, na=False) & series.str.len().le(50)
    description_post = [(None, capitalize_first)]
    # Combine regex validation and max length 50
    location_description_val = lambda series: series.str.match(MY_REGX.LOCATION_DESCRIPTION, na=False) & series.str.len().le(50)
    location_description_post = [(None, capitalize_first)]
    # arrest / domestic fields look to be all True / False, so let's just confirm this with a valid values constraint
    tf_valid = ['true', 'false']
    # beat, district, ward, community area, look to be all integer fields, so for these four, we will 
    # validate it's an int
    valid_int = lambda series: series.str.isdigit()

    # Location, we want to extract lat / lon into their own columns
    def location_lat_lon(df, locations):
        df[['latitude','longitude']] = locations.str.extract(MY_REGX.LOCATION, expand=True)

    # Zip codes need to be stripped of 0 precision (automatic float conversion)
    # Also, we want to prefix 4 length zip codes with a '0' at the beginning after validation
    def zip_to_five(zips):
        condlist = [zips.str.len().eq(4), zips.str.len().eq(5)]
        choicelist = ['0' + zips, zips]
        return pd.Series(index=zips.index, data=np.select(condlist, choicelist, default=np.nan))

    post_zip_codes = [(None, lambda series: zip_to_five(series))]

    nullable_fields = {}
    nullable_fields['block'] = {'validation': MY_REGX.BLOCK, 'generated_cols': [block_num_addr]}
    nullable_fields['iucr'] = {'validation': MY_REGX.IUCR}
    nullable_fields['primary_type'] = {'validation': MY_REGX.PRIMARY_TYPE, 'post_process': primary_type_post}
    nullable_fields['description'] = {'validation': description_val, 'post_process': description_post}
    nullable_fields['location_description'] = {'validation': location_description_val, 'post_process': location_description_post}
    nullable_fields['arrest'] = {'valid_values': tf_valid}
    nullable_fields['domestic'] = {'valid_values': tf_valid}
    nullable_fields['beat'] = {'validation': valid_int}
    nullable_fields['district'] = {'validation': valid_int}
    nullable_fields['ward'] = {'validation': valid_int}
    nullable_fields['community_area'] = {'validation': valid_int}
    nullable_fields['location'] = {'validation': MY_REGX.LOCATION, 'generated_cols': [location_lat_lon]}
    nullable_fields['zip_codes'] = {'validation': MY_REGX.ZIP_CODES, 'post_process': post_zip_codes}

    for field, params in nullable_fields.items():
        logger.debug(f'Processing {field}')
        data_utils.process_field(df_filt, field, soft_rejects, **params)
        
    return soft_rejects
    
