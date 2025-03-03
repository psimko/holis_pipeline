#######################################################################################
#########   File load functions for hicam (.fli) files - Holis project ################ 
#######################################################################################

# (Preprocessing functions are further down)

### MODULES ###

import numpy as np
from ast import literal_eval
import json
from pprint import pprint as print
import os
from numcodecs import Blosc
# from numcodecs import blosc
# blosc.set_nthreads(16)
# from zarr_stores.h5_nested_store import H5_Nested_Store
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
import zarr
from skimage import io
import dask
from dask import delayed

#######################################################################################

### HEADER CONTENT ###

header_info = {
    # '{FLIMIMAGE}'
    # [INFO]
    'version': None,
    'compression': None,
    # '[LAYOUT]'
    'timeStamp': None,
    'CaptureVersion': None,
    'datatype': None,  # Should be UINT12
    'channels': None,
    'x': None,  # X axis pixel dimensions
    'y': None,  # Y axis pixel dimensions
    'z': None,
    'phases': None,
    'frequencies': None,
    'frameRate': None,
    'exposureTime': None,
    'deviceName': None,
    'deviceSerial': None,
    'deviceAlias': None,
    'packing': None,
    'hasDarkImage': None,
    'lutPath': None,
    'timestamps': None,  # Number of frames collected, sometimes does not exist
    # '[DEVICE SETTINGS]'
    'Intensifier_PowerSwitch': None,
    'Intensifier_MCPvoltage': None,
    'Intensifier_MinimumMCPvoltage': None,
    'Intensifier_MaximumMCPvoltage': None,
    'Intensifier_SpecialCharacters': None,
    'Intensifier_AnodeCurrentLevelMicroAmps': None,
    'Intensifier_AnodeCurrentShutdownLevelMicroAmps': None,
    'Intensifier_AnodeCurrentProtectionSwitch': None,
    'Intensifier_UseCustomShutdownAnodeCurrentLevel': None,
    'Intensifier_CloseGateIfCameraIdleAvailable': None,
    'Intensifier_TECtargetTemperature': None,
    'Intensifier_TECheatsinkTemp': None,
    'Intensifier_TECobjectTemp': None,
    'Intensifier_GateOpenSwitch': None,
    'Intensifier_GateOpenTimeSeconds': None,
    'Intensifier_GateDelayTimeSeconds': None,
    'Intensifier_OutputAopenSwitch': None,
    'Intensifier_OutputAopenTimeSeconds': None,
    'Intensifier_OutputAdelayTimeSeconds': None,
    'Intensifier_OutputBopenSwitch': None,
    'Intensifier_OutputBopenTimeSeconds': None,
    'Intensifier_OutputBdelayTimeSeconds': None,
    'Intensifier_SyncMode': None,
    'Intensifier_GatingFixedFrequencyHz': None,
    'Intensifier_OutputAisPolarityPositive': None,
    'Intensifier_GateLoopModeSwitch': None,
    'Intensifier_BurstModeSwitch': None,
    'Intensifier_NumberOfBursts': None,
    'Intensifier_MultipleExposureSwitch': None,
    'Intensifier_NumberOfMultipleExposurePulses': None,
    'Intensifier_GateEnableInputSwitch': None,
    # 'headerLength': header_length,  # Index of last entry in the header
}

#######################################################################################

### .fli TO .omezarr FUNCTION ###

def send_hicam_to_zarr_par_read_once(hicam_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, chunk_lat=128, frames_at_once=1024):
    """
    Function to convert a hicam .fli file into an omezarr file.

    Args:
        hicam .fli file
        output location
        compressor_type='zstd'
        compressor_level=5
        shuffle=1 
        chunk_depth=128 
        chunk_lat=128 
        frames_at_once=1024
    Returns:
        a zarr folder converion of the input file with corresponding zarr specifications (chunk dims, etc.)

    Example:
        spool_files = (
            'wholeScan-run013-z01-y13-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-088.fli',
            'wholeScan-run013-z01-y13-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli',
        )
        spool_files = [os.path.join(base,x) for x in spool_files]
        zarr_locations = [os.path.join(r'/bil/users/psimko/holis/pynet/hicam_data_processing/',os.path.split(x)[-1] + '_ZARR_OUT') for x in spool_files]
        #zarr_locations = [x + '_ZARR_OUT' for x in spool_files]

        def run():
            for spool_file, zarr_location in zip(spool_files, zarr_locations):
                send_hicam_to_zarr_par_read_once(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=128)
        run()
    """

    compressor = Blosc(
        cname=compressor_type,
        clevel=compressor_level,
        shuffle=shuffle,
        blocksize=0
    )

    # zarr_location = os.path.split(spool_file)[0] + '/out_zarr7' ## TEMP FOR TESTING

    # Get Store
    store = H5_Nested_Store(zarr_location)

    # Dump header information to root of zarr store
    header_dict, raw_string = read_header(hicam_file)
    header_json = json.dumps(header_dict, indent=4)
    store['README.txt'] = f'''
        This zarr array was created using holis_tools which turns 12bit hicam camera files into zarr arrays
        In addition to zarr, two plain text files should be found in this folder called: header.json and header_raw.txt

        Origional File: {hicam_file}

        Function used to create this zarr array: holis_tools.send_hicam_to_zarr_par_read_once
        Specific command: send_hicam_to_zarr_par_read_once({hicam_file=},{zarr_location=},{compressor_type=},{compressor_level=},{shuffle=}, {chunk_depth=},{chunk_lat=}, {frames_at_once=})

        header.json: A json representation of data extracted from the hicam file header collected by function holis_tools.hicam_utils.read_header
        header_raw.txt: The RAW header information from the hicam file dumped here.
        '''.encode()
    store['header.json'] = header_json.encode()
    store['header_raw.txt'] = raw_string.encode()

    num_frames = get_number_of_frames(hicam_file, header_info=header_dict)
    frame_shape = get_frame_shape(hicam_file, header_info=header_dict)

    array_shape = (int(num_frames), int(frame_shape[0]), int(frame_shape[1]))
    # chunks = (chunk_depth, array_shape[1] // 2,
    #           array_shape[2] // 2)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame
    chunks = (chunk_depth, chunk_lat,
              chunk_lat)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame

    array = zarr.zeros(store=store, shape=array_shape, chunks=chunks, compressor=compressor,
                       dtype='uint16')

    # header_info, _ = read_header(file_name)
    # header_len = header_info['headerLength']

    print('Reading hicam file into memory')

    with open(hicam_file, 'rb') as f:

        to_process = []
        for location in get_start_stop_reads_for_frame_groups(hicam_file, header_info=None, frames_at_once=frames_at_once):

            f.seek(location['start'])
            frames = f.read(location['len'])

            #queue to write to zarr
            tmp = delayed(write_part_bytes)(zarr_location, frames, location)
            to_process.append(tmp)
            del tmp

    out = dask.compute(to_process)

    # from dask.distributed import Client
    # with Client() as client:
    #     print('Computing')
    #     out = client.compute(to_process)

#######################################################################################

### HELPER FUNCTIONS


def get_header_size(file_name, header_info=None):

    if header_info is None:
        header_info, _ = read_header(file_name)

    return header_info['headerLength']

def get_frame_shape(file_name, header_info=None):

    if header_info is None:
        header_info, _ = read_header(file_name)

    return (header_info['y'], header_info['x'])


def get_number_of_frames(file_name, header_info=None):

    if header_info is None:
        header_info, _ = read_header(file_name)

    pixelInFrame_bit8 = int(header_info['x'] * header_info['y'] / 2 * 3)  # Number of bits in frame

    how_many_frames = header_info['timestamps']
    if how_many_frames is None:
        with open(file_name, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size_of_file = f.tell()
            header_len = header_info['headerLength']
            data_size = size_of_file - header_len
            num_frames_remainder = data_size % pixelInFrame_bit8
            assert num_frames_remainder == 0, 'The length of the spool file does not fit an integer number of frames'
            how_many_frames = data_size // pixelInFrame_bit8

    return how_many_frames

def get_hicam_zarr(zarr_location, mode='r'):
    store = H5_Nested_Store(zarr_location, 'r')
    array = zarr.open(store,mode)
    print(array)
    return array

def read_uint12(data_chunk, coerce_to_uint16_values=True):
    '''
    Since numpy does not understand uint12 data, this function takes a raw bytes object and reads the 12bit integer
    data into a uint16 array.

    Input:
        data_chunk: Byte string
        coerce_to_uint16_values (bool): if True outputs an array with values that are scaled to uint16

        In general this should remain True.  Thus, the image is representative of a conversion to UINT16 precision
        and any conversion to other precision images (ie float for processing) will appropriately represent the
        original data. *Manual conversion to the original uint12 values can be obtained by division by 16

    Output:
         uint16 numpy array where each integer corresponds to the uint12 value (default)

    '''

    array = read_uint12_c(data_chunk)
    if coerce_to_uint16_values:
        array *= 16
    return array

def write_part_bytes(zarr_location, bytes_from_file, writedict):
    array = get_hicam_zarr(zarr_location, mode='a')

    print(f'Forming Array')
    chunk_shape = (
        writedict['frames'],
        array.shape[1],
        array.shape[2]
    )
    output = np.zeros(chunk_shape, 'uint16')
    for idx in range(writedict['frames']):
        where_to_start = idx * writedict['pixelInFrame_bit8']
        data = bytes_from_file[where_to_start:where_to_start + writedict['pixelInFrame_bit8']]

        # Data to uint16 where uint12 values have been scaled to uint16 values
        # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
        canvas = read_uint12(data, coerce_to_uint16_values=True)
        # canvas = read_uint12(data, coerce_to_uint16_values=False)

        output[idx] = canvas.reshape((header_info['y'], header_info['x']))

    start = writedict['group'] * writedict['frames_at_once']
    stop = start + writedict['frames']
    array[start:stop] = output
    del array

#######################################################################################

### READ HEADER FUNCTION ###

def read_header(file_name):
    '''
    Given a file name, read hicam header and extract parameters into a dictionary.  Make an effort to coerce
    values into appropriate types.

    Return dictionary

    Dictionary includes an extra key 'headerLength'.  All data after this index is image frame data.
    '''
    read_n = 0
    fileinfo = b''
    with open(file_name, 'rb') as f:
        while read_n < 100:
            a = f.read(1000)
            fileinfo += a
            if b'{END}' in fileinfo:
                header_length = fileinfo.index(b'{END}') + 5
                # location = str(fileinfo).index('{END}')
                print(f'Found the end of header at location {header_length}')

                # Convert fileinfo to a string and trim b' and remove anything after {END}
                header_string_end = str(fileinfo).index('{END}') + 5
                fileinfo = str(fileinfo)[2:header_string_end]
                break
            read_n += 1
        if read_n == 100:
            raise KeyError('Error while reading HICAM Header, Header Length too long?')

    raw_header_string = fileinfo

    # Edit: change fileinfo to raw header string
    
    # Extract header info
    fileinfo = raw_header_string.split('\\n')     
    # for idx, ii in enumerate(fileinfo):
        # print(ii)

    for ii in fileinfo:
        for key in header_info:
            test = key.lower() + ' = '
            if ii.lower().startswith(test):
                header_info[key] = ii[len(test)::]

    # Automatically convert values to appropriate types
    for key, value in header_info.items():
        try:
            header_info[key] = literal_eval(value)
        except Exception:
            pass
    header_info['headerLength'] = header_length
    # print(header_info)
    return header_info, raw_header_string

#######################################################################################

def read_uint12_c(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = ((mid_uint8 & 0x0F) << 8) | fst_uint8
    snd_uint12 = (lst_uint8 << 4) | ((mid_uint8 & 0xF0) >> 4)
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    return array

def read_data_file(spool_file, header_info=None):

    if header_info is None:
        header_info, _ = read_header(spool_file)

    ################################################################
    ## READ HICAM DATA IN ALL AT ONCE then convert to numpy z-stack
    ################################################################

    pixelInFrame_bit8 = int(int(header_info['x']) * int(header_info['y']) / 2 * 3)  # Number of bits in frame

    how_many_frames = header_info['timestamps']
    if how_many_frames is None:
        with open(spool_file, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size_of_file = f.tell()
            # print(f'{size_of_file=}')
            header_len = header_info['headerLength']
            # print(f'{header_len=}')
            data_size = size_of_file - header_len
            # print(f'{data_size=}')
            num_frames_remainder = data_size%pixelInFrame_bit8
            # print(f'{num_frames_remainder=}')
            how_many_frames = int(data_size//pixelInFrame_bit8)
            # print(f'{how_many_frames=}')

    chunk_shape = (int(how_many_frames), int(header_info['y']), int(header_info['x']))

    output = np.zeros(chunk_shape, 'uint16')

    #make tuple of slices to extract

    with open(spool_file, 'rb') as f:
        size_of_file = f.tell()
        # print(f'{size_of_file=}')
        # print(f'Reading {how_many_frames} frames')
        f.seek(header_info['headerLength'])
        multi_frame = f.read(int(how_many_frames) * int(pixelInFrame_bit8))

    print(f'Forming Array')
    for idx in range(int(how_many_frames)):
        # print(f'Converting {idx} of {how_many_frames}')
        where_to_start = idx * pixelInFrame_bit8
        data = multi_frame[where_to_start:where_to_start + pixelInFrame_bit8]

        # Data to uint16 where uint12 values have been scaled to uint16 values
        # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
        canvas = read_uint12_c(data) # removed 'coerce_to_uint16_values=True' parameter
        # canvas = read_uint12(data, coerce_to_uint16_values=False)

        output[idx] = canvas.reshape((int(header_info['y']), int(header_info['x'])))

    return output

#######################################################################################

### GENERATE START AND STOP FRAMES ###

def get_start_stop_reads_for_frame_groups(file_name, header_info=None, frames_at_once=1024):
    '''
    Given a file name, read hicam header and extract start and stop indexes of groups of frames.

    Return generator of dictionaries
    '''
    if header_info is None:
        header_info, _ = read_header(file_name)

    pixelInFrame_bit8 = int(header_info['x'] * header_info['y'] / 2 * 3)  # Number of bits in frame

    how_many_frames = header_info['timestamps']
    if how_many_frames is None:
        with open(file_name, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size_of_file = f.tell()
            header_len = header_info['headerLength']
            data_size = size_of_file - header_len
            num_frames_remainder = data_size % pixelInFrame_bit8
            assert num_frames_remainder == 0, 'The length of the spool file does not fit an integer number of frames'
            how_many_frames = data_size // pixelInFrame_bit8
            
    else:
        data_size = how_many_frames * pixelInFrame_bit8
        header_len = header_info['headerLength']

    read_len = frames_at_once * pixelInFrame_bit8
    remaining = data_size
    start = header_len
    idx = 0
    while remaining > 0:
        if remaining - read_len < 0:
            stop = start + remaining
            remaining = 0
        else:
            stop = start + read_len
            remaining -= read_len

        length = stop-start
        yield {'start':start,
               'stop':stop,
               'group':idx,
               'len':length,
               'file':file_name,
               'frames':length//pixelInFrame_bit8,
               'pixelInFrame_bit8':pixelInFrame_bit8,
               'last':remaining==0,
               'frames_at_once':frames_at_once}
        start = stop
        idx += 1

#######################################################################################





