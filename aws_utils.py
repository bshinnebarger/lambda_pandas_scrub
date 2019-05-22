import logging
import re
import pathlib
import s3fs

s3_logger = logging.getLogger('main.s3_utils')

                                                
def get_s3_files_to_process(fs, file_pat, bucket, prefix=''):
    '''
    Get objects in an s3 bucket matching a regex file pattern and optional prefix
    
    Parameters:
    fs (s3fs.S3FileSystem): s3fs file system
    file_pat (re Pattern): regex pattern with a group capture for sorting
    bucket (str): s3 bucket name
    
    Keyword arguments:
    prefix (str): s3 "folder" prefix
    
    Returns:
    files (list of str): list of full paths to s3 objects that match the conditions, sorted by the regex capture group
    '''
    try:
        bucket_contents = fs.ls(bucket+'/'+prefix)
    except FileNotFoundError as err:
        s3_logger.error(f'Error while trying to retrieve contents of {bucket}/{prefix}')
        raise
        
    files = []

    s3_logger.info(f'Found {len(bucket_contents)} files to process')
    for bucket_item in bucket_contents:
        file_num_search = re.search(file_pat, bucket_item)

        if file_num_search:
            file = file_num_search.group(0)
            file_num = file_num_search.group(1)
            files.append((file_num, file))

    if len(files) > 0 and files[0][0].isdigit():
        files = sorted(files, key=lambda x: int(x[0])) 
    else:
        files = sorted(files, key=lambda x: x[0]) 
                     
    return files
                
                        
def download_s3_file(fs, s3_bucket_key, destination):
    '''
    Download s3 object to local destination
    
    Parameters:
    fs (s3fs.S3FileSystem): s3fs file system
    s3_bucket_key (str): full path the s3 object
    destination (pathlib Path): path to download destination file
    '''
    if destination.exists():
        s3_logger.info(f'File {destination.as_posix()} already exists')
        return
    else:
        if not destination.parent.exists():
            s3_logger.info(f'Destination {destination.parent.as_posix()} doesn"t exist, creating it')
            destination.parent.mkdir(parents=True, exist_ok=True)    
    
    try:
        fs.get(s3_bucket_key, destination)
    except FileNotFoundError as err:
        s3_logger.error(f'Error while trying to download {s3_bucket_key}')
        raise
                        
    s3_logger.info(f'Successfully downloaded {s3_bucket_key}')

                        
def upload_s3_file(fs, local_file, s3_bucket_key):
    '''
    Upload local file to s3
    
    Parameters:
    fs (s3fs.S3FileSystem): s3fs file system
    local_file (pathlib Path): path of local file to upload
    s3_bucket_key (str): full path the s3 object
    '''
    if not local_file.exists():
        s3_logger.error(f'File {local_file.as_posix()} doesn"t exists')
        raise FileNotFoundError(f'Local File {local_file.as_posix()} not found')
    
    fs.put(local_file, s3_bucket_key)
    s3_logger.info(f'Successfully uploaded {local_file.name} to {s3_bucket_key}')
              
