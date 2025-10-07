# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:45:45 2022

@author: alpha
"""

'''
This module provides tools to read hicam data files and convert the data to compressed zarr arrays 

hicam data is saved as UINT12.  A function converts UINT12>>UINT16 then appropriately rescales the data to
UINT16 values. 0:4095 >> 0:65534.  Data are stored as UINT16. To recover the original UINT12 values simply divide by 16

Arrays are saved according to hicam format: thus axes x,y,and z are representative of the cameras coordinates.  
X and Y are lateral axes (each 2D frame capture), Z represents a stack of many frames.

SCAPE data acquisition often requires that these axes are rearranged to represent acquisition axes.  For example a
mapping of hicam axes to SCAPE acquisition axes would be:
{
# hicam:scape #
'y':'z',
'x':'y',
'z':'x'
}
'''

import numpy as np
from ast import literal_eval
import json
# from pprint import pprint as print
import os
from numcodecs import Blosc
# from numcodecs import blosc
# blosc.set_nthreads(16)
# from zarr_stores.h5_nested_store import H5_Nested_Store
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
import zarr
import zstandard as zstd
from skimage import io

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


def is_compressed_fli(file_name):
    # Extract the first extension (.zst)
    file_name, ext1 = os.path.splitext(file_name)

    if ext1.lower() == '.fli':
        return False

    # Extract the second extension (.fli) from the remaining filename
    _, ext2 = os.path.splitext(file_name)

    assert ext2.lower() == '.fli', 'File does not appear to be a compressed fli.'
    return ext1.lower() == '.zst'


def get_len_fli(file_name):
    compressed = is_compressed_fli(file_name)
    if not compressed:
        return os.path.getsize(file_path)
    else:
        with open(file_name, 'rb') as f:
            header_size = 18
            header_data = f.read(header_size)

            # Parse the header to get frame parameters
            frame_params = zstd.get_frame_parameters(header_data)

            # The content_size attribute holds the decompressed size
            # A value of -1 means the size is not stored in the header
            if frame_params.content_size >= 0:
                return frame_params.content_size

    return None


import zstandard as zstd
from pathlib import Path


class FliOpen:
    """
    A context manager for opening files, handling .fli.zstd compression automatically.

    If the file ends with '.zstd', it returns a zstandard decompressor stream reader.
    Otherwise, it returns a standard file object from open().
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.compressed = is_compressed_fli(file_name)
        # self.size = get_len_fli(file_name)
        self._f_in = None
        self._file_obj = None

    def __enter__(self):
        # Open the file in binary mode for both compressed and uncompressed files
        # because zstandard works with binary streams.
        # The mode is adjusted inside the decompression logic.
        self._f_in = open(self.file_name, 'rb')

        # Check if the file has a .zstd extension
        if self.compressed:
            dctx = zstd.ZstdDecompressor()
            # Wrap the binary file handle in a zstandard stream reader context manager.
            # This is also a context manager, so we store the object it returns.
            self._file_obj = dctx.stream_reader(self._f_in)
            return self._file_obj
        else:
            # For non-zstd files, use the regular open() file handle.
            self._file_obj = self._f_in
            return self._file_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure all open file handles are properly closed upon exiting the context.
        if self._file_obj:
            self._file_obj.close()
        if self._f_in:
            self._f_in.close()
        return False  # Propagate any exceptions


def read_header(file_name):
    '''
    Given a file name, read hicam header and extract parameters into a dictionary.  Make an effort to coerce
    values into appropriate types.

    Return dictionary

    Dictionary includes an extra key 'headerLength'.  All data after this index is image frame data.
    '''
    read_n = 0
    fileinfo = b''
    with FliOpen(file_name) as f:
        while read_n < 40:
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
        if read_n == 40:
            raise KeyError('Error while reading HICAM Header, Header Length too long?')

    raw_header_string = fileinfo
    # Extract header info
    fileinfo = fileinfo.split('\\n')
    for idx, ii in enumerate(fileinfo):
        print(ii)

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
    print(header_info)
    return header_info, raw_header_string


def read_uint12_a(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    return array


def read_uint12_b(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = (lst_uint8 << 4) + (np.bitwise_and(15, mid_uint8))
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    return array


### THIS ONE WORKS FOR HICAM ###
def read_uint12_c(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = ((mid_uint8 & 0x0F) << 8) | fst_uint8
    snd_uint12 = (lst_uint8 << 4) | ((mid_uint8 & 0xF0) >> 4)
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
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


def read_data_file(spool_file, header_info=None):
    if header_info is None:
        header_info, _ = read_header(spool_file)

    ################################################################
    ## READ HICAM DATA IN ALL AT ONCE then convert to numpy z-stack
    ################################################################

    pixelInFrame_bit8 = int(header_info['x'] * header_info['y'] / 2 * 3)  # Number of bits in frame

    size_of_file = get_len_fli(spool_file)

    how_many_frames = header_info['timestamps']
    if how_many_frames is None:
        print(f'{size_of_file=}')
        header_len = header_info['headerLength']
        print(f'{header_len=}')
        data_size = size_of_file - header_len
        print(f'{data_size=}')
        num_frames_remainder = data_size % pixelInFrame_bit8
        print(f'{num_frames_remainder=}')
        how_many_frames = data_size // pixelInFrame_bit8
        print(f'{how_many_frames=}')

    chunk_shape = (how_many_frames, header_info['y'], header_info['x'])

    output = np.zeros(chunk_shape, 'uint16')

    # make tuple of slices to extract

    with open(spool_file, 'rb') as f:
        # size_of_file = f.tell()
        print(f'{size_of_file=}')
        print(f'Reading {how_many_frames} frames')
        f.seek(header_info['headerLength'])
        multi_frame = f.read(how_many_frames * pixelInFrame_bit8)

    print(f'Forming Array')
    for idx in range(how_many_frames):
        print(f'Converting {idx} of {how_many_frames}')
        where_to_start = idx * pixelInFrame_bit8
        data = multi_frame[where_to_start:where_to_start + pixelInFrame_bit8]

        # Data to uint16 where uint12 values have been scaled to uint16 values
        # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
        canvas = read_uint12(data, coerce_to_uint16_values=True)
        # canvas = read_uint12(data, coerce_to_uint16_values=False)

        output[idx] = canvas.reshape((header_info['y'], header_info['x']))

    return output


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
        size_of_file = get_len_fli(file_name)
        header_len = header_info['headerLength']
        data_size = size_of_file - header_len
        num_frames_remainder = data_size % pixelInFrame_bit8
        assert num_frames_remainder == 0, 'The length of the spool file does not fit an integer number of frames'
        how_many_frames = data_size // pixelInFrame_bit8

    return how_many_frames


def read_part_data_file(file_name, header_info=None, frames_at_once=1024):
    if header_info is None:
        header_info, _ = read_header(file_name)

    ################################################################
    ## READ HICAM DATA IN ALL AT ONCE then convert to numpy z-stack
    ################################################################

    pixelInFrame_bit8 = int(header_info['x'] * header_info['y'] / 2 * 3)  # Number of bits in frame

    chunk_shape = (frames_at_once, header_info['y'], header_info['x'])

    output = np.zeros(chunk_shape, 'uint16')

    start_index = get_header_size(file_name, header_info=header_info)
    read_len = frames_at_once * pixelInFrame_bit8

    with FliOpen(file_name) as f:
        f.seek(start_index)

        read_number = 0
        while True:
            multi_frame = f.read(read_len)

            if len(multi_frame) == 0:
                '''When no more data remains exit the while loop'''
                break

            current_num_frames = len(multi_frame) // pixelInFrame_bit8

            read_number += 1
            print(f'Reading frames set {read_number} of {current_num_frames} frames')

            # print(f'Forming Array')
            for idx in range(current_num_frames):
                where_to_start = idx * pixelInFrame_bit8
                data = multi_frame[where_to_start:where_to_start + pixelInFrame_bit8]
                # print(f'{current_num_frames=}, {where_to_start=}, {idx=}, Data length = {len(data)}')

                # Data to uint16 where uint12 values have been scaled to uint16 values
                # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
                canvas = read_uint12(data, coerce_to_uint16_values=True)
                # print(f'{canvas.shape}')
                # canvas = read_uint12(data, coerce_to_uint16_values=False)

                output[idx] = canvas.reshape((header_info['y'], header_info['x']))

            yield output[:current_num_frames]


'''
Code below is designed to package a hicam spool file as a zarr
'''


def header_info_to_json(json_file, header_info):
    with open(json_file, 'w') as f:
        f.write(json.dumps(header_info, indent=4))


def send_hicam_to_zarr(hicam_file, zarr_location, compressor_type='zstd', compressor_level=5, shuffle=1,
                       chunk_depth=128, frames_at_once=1024):
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
    store['header.json'] = header_json.encode()
    store['header_raw.txt'] = raw_string.encode()

    num_frames = get_number_of_frames(hicam_file, header_info=header_dict)
    frame_shape = get_frame_shape(hicam_file, header_info=header_dict)

    array_shape = (num_frames, frame_shape[0], frame_shape[1])
    chunks = (chunk_depth, array_shape[1] // 2,
              array_shape[2] // 2)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame

    array = zarr.zeros(store=store, shape=array_shape, chunks=chunks, compressor=compressor,
                       dtype='uint16')

    # image_data = read_data_file(hicam_file, header_info=None)
    # image_data = read_part_data_file(hicam_file, header_info=None)
    for idx, ii in enumerate(read_part_data_file(hicam_file, header_info=None, frames_at_once=frames_at_once)):
        start = idx * frames_at_once
        stop = start + ii.shape[0]
        print(ii.shape)

        print('Writing to ZARR')
        array[start:stop] = ii


def get_start_stop_reads_for_frame_groups(file_name, header_info=None, frames_at_once=1024):
    if header_info is None:
        header_info, _ = read_header(file_name)

    pixelInFrame_bit8 = int(header_info['x'] * header_info['y'] / 2 * 3)  # Number of bits in frame

    # how_many_frames = header_info['timestamps']
    size_of_file = get_len_fli(file_name)
    header_len = header_info['headerLength']
    data_size = size_of_file - header_len
    # if how_many_frames is None:
    #     num_frames_remainder = data_size % pixelInFrame_bit8
    #     assert num_frames_remainder == 0, 'The length of the spool file does not fit an integer number of frames'
    #     how_many_frames = data_size // pixelInFrame_bit8

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

        length = stop - start
        yield {'start': start,
               'stop': stop,
               'group': idx,
               'len': length,
               'file': file_name,
               'frames': length // pixelInFrame_bit8,
               'pixelInFrame_bit8': pixelInFrame_bit8,
               'last': remaining == 0,
               'frames_at_once': frames_at_once}
        start = stop
        idx += 1


def write_part(zarr_location, writedict):
    array = get_hicam_zarr(zarr_location, mode='a')
    with open(writedict['file'], 'rb') as f:
        f.seek(writedict['start'])
        multi_frame = f.read(writedict['len'])

    print(f'Forming Array')
    chunk_shape = (
        writedict['frames'],
        array.shape[1],
        array.shape[2]
    )
    output = np.zeros(chunk_shape, 'uint16')
    for idx in range(writedict['frames']):
        where_to_start = idx * writedict['pixelInFrame_bit8']
        data = multi_frame[where_to_start:where_to_start + writedict['pixelInFrame_bit8']]

        # Data to uint16 where uint12 values have been scaled to uint16 values
        # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
        canvas = read_uint12(data, coerce_to_uint16_values=True)
        # canvas = read_uint12(data, coerce_to_uint16_values=False)

        output[idx] = canvas.reshape((header_info['y'], header_info['x']))

    start = writedict['group'] * writedict['frames_at_once']
    stop = start + writedict['frames']
    array[start:stop] = output
    del array


def send_hicam_to_zarr_par(hicam_file, zarr_location, compressor_type='zstd', compressor_level=5, shuffle=1,
                           chunk_depth=128, frames_at_once=1024, cube_chunk=None):
    import dask
    from dask import delayed
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

    Function used to create this zarr array: holis_tools.hicam_utils.send_hicam_to_zarr_par
    Specific command: send_hicam_to_zarr_par({hicam_file=},{zarr_location=},{compressor_type=},{compressor_level=},{shuffle=}, {chunk_depth=}, {frames_at_once=}, {cube_chunk=})

    header.json: A json representation of data extracted from the hicam file header collected by function holis_tools.hicam_utils.read_header
    header_raw.txt: The RAW header information from the hicam file dumped here.
    '''.encode()
    store['header.json'] = header_json.encode()
    store['header_raw.txt'] = raw_string.encode()

    num_frames = get_number_of_frames(hicam_file, header_info=header_dict)
    frame_shape = get_frame_shape(hicam_file, header_info=header_dict)

    array_shape = (num_frames, frame_shape[0], frame_shape[1])
    # chunks = (chunk_depth, array_shape[1] // 2, array_shape[2] // 2) #Chunks in x,y are 2x2 to account for the 2x2 channels in each frame

    if cube_chunk:
        chunks = (cube_chunk, cube_chunk, cube_chunk)
    else:
        chunks = (chunk_depth, array_shape[1] // 4,
                  array_shape[2] // 4)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame

    array = zarr.zeros(store=store, shape=array_shape, chunks=chunks, compressor=compressor,
                       dtype='uint16')

    to_process = []
    for write in get_start_stop_reads_for_frame_groups(hicam_file, header_info=None, frames_at_once=frames_at_once):
        tmp = delayed(write_part)(zarr_location, write)
        to_process.append(tmp)
        del tmp

    print('Converting hicam file')
    dask.compute(to_process)


####################################################################################
'''Read all data in first then write to zarr in parallel'''


####################################################################################

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
    try:
        array[start:stop] = output
    except:
        print(">>>>>>>>>>>Exception - last chunk")
        print(">>>>>>>>>>>array[start:stop].shape", array[start:stop].shape)
        print(">>>>>>>>>>>output.shape", output.shape)
        to_read = stop - start - 1
        print(">>>>>>>>>>>to_read", to_read)
        array[start:stop] = output[:to_read]
    del array


# tmp = delayed(write_part_bytes_all)(zarr_location, frames, location, hicam_file, location['start'], location['len'])
def write_part_bytes_all(zarr_location, writedict, file, start, stop):
    with FliOpen(file, 'rb') as f:
        f.seek(start)
        bytes_from_file = f.read(stop)

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


def send_hicam_to_zarr_par_read_once(hicam_file, zarr_location, compressor_type='zstd', compressor_level=5, shuffle=1,
                                     chunk_depth=128, chunk_lat=128, frames_at_once=1024):
    import dask
    from dask import delayed

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

        Original File: {hicam_file}

        Function used to create this zarr array: holis_tools.send_hicam_to_zarr_par_read_once
        Specific command: send_hicam_to_zarr_par_read_once({hicam_file=},{zarr_location=},{compressor_type=},{compressor_level=},{shuffle=}, {chunk_depth=},{chunk_lat=}, {frames_at_once=})

        header.json: A json representation of data extracted from the hicam file header collected by function holis_tools.hicam_utils.read_header
        header_raw.txt: The RAW header information from the hicam file dumped here.
        '''.encode()
    store['header.json'] = header_json.encode()
    store['header_raw.txt'] = raw_string.encode()

    print("Read header")

    num_frames = get_number_of_frames(hicam_file, header_info=header_dict)
    print("num_frames", num_frames)
    frame_shape = get_frame_shape(hicam_file, header_info=header_dict)
    print("frame_shape", frame_shape)

    array_shape = (num_frames, frame_shape[0], frame_shape[1])
    print("array_shape", array_shape)
    # chunks = (chunk_depth, array_shape[1] // 2,
    #           array_shape[2] // 2)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame
    chunks = (chunk_depth, chunk_lat,
              chunk_lat)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame
    print("chunks", chunks)
    array = zarr.zeros(store=store, shape=array_shape, chunks=chunks, compressor=compressor,
                       dtype='uint16')
    print("array", array)

    # header_info, _ = read_header(file_name)
    # header_len = header_info['headerLength']

    print('Reading hicam file into memory')

    with FliOpen(hicam_file) as f:
        to_process = []
        for location in get_start_stop_reads_for_frame_groups(hicam_file, header_info=None,
                                                              frames_at_once=frames_at_once):
            f.seek(location['start'])
            frames = f.read(location['len'])

            # queue to write to zarr
            tmp = delayed(write_part_bytes)(zarr_location, frames, location)
            to_process.append(tmp)
            del tmp

    out = dask.compute(to_process)

    # from dask.distributed import Client
    # with Client() as client:
    #     print('Computing')
    #     out = client.compute(to_process)


def send_hicam_to_zarr_par_read_groups(hicam_file, zarr_location, compressor_type='zstd', compressor_level=5, shuffle=1,
                                       chunk_depth=128, chunk_lat='quarters', frames_at_once=1024, groups=100):
    import dask
    from dask import delayed

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

        Function used to create this zarr array: holis_tools.hicam_utils.ssend_hicam_to_zarr_par_read_groups
        Specific command: send_hicam_to_zarr_par_read_groups({hicam_file=},{zarr_location=},{compressor_type=},{compressor_level=},{shuffle=}, {chunk_depth=}, {chunk_lat=}, {frames_at_once=}, {groups=})

        header.json: A json representation of data extracted from the hicam file header collected by function holis_tools.hicam_utils.read_header
        header_raw.txt: The RAW header information from the hicam file dumped here.
        '''.encode()
    store['header.json'] = header_json.encode()
    store['header_raw.txt'] = raw_string.encode()

    num_frames = get_number_of_frames(hicam_file, header_info=header_dict)
    frame_shape = get_frame_shape(hicam_file, header_info=header_dict)

    array_shape = (num_frames, frame_shape[0], frame_shape[1])
    # chunks = (chunk_depth, array_shape[1] // 2,
    #           array_shape[2] // 2)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame
    if chunk_lat == 'quarters':
        chunks = (chunk_depth, array_shape[1] // 2, array_shape[2] // 2)
    else:
        chunks = (chunk_depth, chunk_lat,
                  chunk_lat)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame

    array = zarr.zeros(store=store, shape=array_shape, chunks=chunks, compressor=compressor,
                       dtype='uint16')

    # header_info, _ = read_header(file_name)
    # header_len = header_info['headerLength']

    print('Reading hicam file into memory')

    with open(hicam_file, 'rb') as f:

        to_process = []
        on_group = 0
        for location in get_start_stop_reads_for_frame_groups(hicam_file, header_info=None,
                                                              frames_at_once=frames_at_once):

            f.seek(location['start'])
            frames = f.read(location['len'])

            # queue to write to zarr
            tmp = delayed(write_part_bytes)(zarr_location, frames, location)
            to_process.append(tmp)
            del tmp

            on_group += 1
            if on_group == groups:
                print('Computing Group')
                out = dask.compute(to_process)
                on_group = 0
                del to_process
                to_process = []
                print('Reading next Group')

    if len(to_process) > 0:
        print('Computing Final Group')
        out = dask.compute(to_process)
        to_process = []
    print('Computing Completed')

    # from dask.distributed import Client
    # with Client() as client:
    #     print('Computing')
    #     out = client.compute(to_process)


def read_part(file, start, len):
    with open(file, 'rb') as f:
        f.seek(start)
        return f.read(len)


def send_hicam_to_zarr_par_read_groups_par(hicam_file, zarr_location, compressor_type='zstd', compressor_level=5,
                                           shuffle=1, chunk_depth=128, chunk_lat='quarters', frames_at_once=1024,
                                           groups=100):
    import dask
    from dask import delayed
    from distributed import Client, get_client
    client = get_client()

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

            Function used to create this zarr array: holis_tools.hicam_utils.send_hicam_to_zarr_par_read_groups_par
            Specific command: send_hicam_to_zarr_par_read_groups_par({hicam_file=},{zarr_location=},{compressor_type=},{compressor_level=},{shuffle=}, {chunk_depth=}, {chunk_lat=}, {frames_at_once=}, {groups=})

            header.json: A json representation of data extracted from the hicam file header collected by function holis_tools.hicam_utils.read_header
            header_raw.txt: The RAW header information from the hicam file dumped here.
            '''.encode()
    store['header.json'] = header_json.encode()
    store['header_raw.txt'] = raw_string.encode()

    num_frames = get_number_of_frames(hicam_file, header_info=header_dict)
    frame_shape = get_frame_shape(hicam_file, header_info=header_dict)

    array_shape = (num_frames, frame_shape[0], frame_shape[1])

    if chunk_lat == 'quarters':
        chunks = (chunk_depth, array_shape[1] // 2,
                  array_shape[2] // 2)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame
    else:
        chunks = (chunk_depth, chunk_lat,
                  chunk_lat)  # Chunks in x,y are 2x2 to account for the 2x2 channels in each frame

    array = zarr.zeros(store=store, shape=array_shape, chunks=chunks, compressor=compressor,
                       dtype='uint16')

    # header_info, _ = read_header(file_name)
    # header_len = header_info['headerLength']

    print('Reading hicam file into memory')

    to_process = []
    on_group = 0
    for location in get_start_stop_reads_for_frame_groups(hicam_file, header_info=None, frames_at_once=frames_at_once):
        # frames = delayed(read_part)(hicam_file, location['start'], location['len'])

        # queue to write to zarr
        tmp = delayed(write_part_bytes_all)(zarr_location, location, hicam_file, location['start'], location['len'])
        to_process.append(tmp)
        del tmp

        # on_group += 1
        # if on_group == groups:
        #     print('Computing Group')
        #     out = dask.compute(to_process)
        #     on_group = 0
        #     del to_process
        #     to_process = []
        #     print('Reading next Group')

    if len(to_process) > 0:
        print('Computing Final Group')
        # out = dask.compute(to_process)
        # with Client() as client:
        #     out = client.compute(to_process, sync=True)
        # out = client.compute(to_process, sync=True)
        out = client.compute(to_process)
        to_process = []
    print('Computing Completed')
    return out


def get_hicam_zarr(zarr_location, mode='r'):
    store = H5_Nested_Store(zarr_location, 'r')
    array = zarr.open(store, mode)
    print(array)
    return array


class open_hicam_array_as_aligned_color_dataset:
    '''
    Present the hicam data (z,y,x) as a multicolor 3D array.  The class handles all the trimming and alignment of
    channels in (y,x)

    Input
    -----> X (hicam axis)
    --------------------- '
    '    c0   '    c1   ' '
    '         '         ' V
    ---------------------
    '    c2   '    c3   ' Y (hicam axis)
    '         '         '
    ---------------------

    Output
    -----------
    ' c0,1,2,3'
    '         '
    -----------

    '''

    def __init__(self, zarr_location):

        # Store origional hicam_file zarr representation
        self.path = zarr_location
        self.store = H5_Nested_Store(zarr_location, 'r')
        self.array = zarr.open(self.store, 'r')

        # Store details of virtual multicolor array assuming each z-layer is divided into a 2x2 grid of images
        # representing each color
        self.shape = (4, self.array.shape[0], self.array.shape[1] // 2, self.array.shape[2] // 2)
        self.dtype = self.array.dtype
        self.size = self.array.size
        self.ndims = 4
        self.chunks = (4, self.array[0], self.shape[2], self.shape[3])

        self.hicam_color_shifts = self.array.attrs.get('hicam_color_shifts')
        self.lighting_mean = None
        self.lighting_mask = None
        if os.path.exists(os.path.join(self.path, 'mean.tiff')):
            self.lighting_mean = io.imread(os.path.join(self.path, 'mean.tiff'))
            self.lighting_mean = img_as_float32(self.lighting_mean)
            self.lighting_max = self.lighting_mean.max()

    def lighting_correction(self, image):
        image_max = image.max()
        image_32 = img_as_float32(image)
        correct = image_32 / self.lighting_mean
        scale_factor = image_max / correct.max()
        correct *= scale_factor
        print(f'{correct.max()}, {correct.min()}')
        return correct.astype(np.uint16)

    def calculate_shifts(self, z_range=None):
        if z_range is None:
            # Select the middle 1000 z layers to calculate shifts.
            # Only 1 in every 10 frames is calculated, so middle 1000 = 100 frames aligned.
            mid_z = self.shape[1] // 2
            z_range = (mid_z - 500, mid_z + 500)
            # z_range = (mid_z - 1000, mid_z + 1000)
            # z_range = (mid_z - 2500, mid_z + 2500)
        self.hicam_color_shifts = calculate_channel_shifts(self, anchor_channel=0, z_range=z_range)
        # Store origional hicam_file zarr representation
        store = H5_Nested_Store(self.path, 'a')
        array = zarr.open(store, 'a')
        array.attrs['hicam_color_shifts'] = self.hicam_color_shifts

        del array
        del store

    def __getitem__(self, item):
        '''
        Get slice information and apply this to each of the 4 colors quadrants (colors) and combine into
        a single color image.
        '''
        print(f'In slice: {item}')

        if isinstance(item, int):
            item = (slice(item, item + 1), slice(0, self.shape[1]), slice(0, self.shape[2]), slice(0, self.shape[3]))

        elif isinstance(item, slice):
            item = (item, slice(0, self.shape[1]), slice(0, self.shape[2]), slice(0, self.shape[3]))

        elif isinstance(item, tuple):

            tmp_slice = []

            for idx, ii in enumerate(item):
                if isinstance(ii, int):
                    tmp_slice.append(slice(ii, ii + 1))
                elif isinstance(ii, slice):
                    tmp_slice.append(ii)

            idx += 1
            while idx < self.ndims:
                tmp_slice.append(slice(0, self.shape[idx]))
                idx += 1
                # print(idx)

            item = tmp_slice

        for idx in range(self.ndims):

            if item[idx] == slice(None):
                item[idx] = slice(0, self.shape[idx])

            elif isinstance(item[idx], int):
                item[idx] = slice(item[idx], item[idx] + 1)

            elif isinstance(item[idx], slice) and \
                    item[idx].start is None and \
                    isinstance(item[idx].stop, int):
                item[idx] = slice(item[idx].stop, item[idx].stop + 1)

        for idx, slc in enumerate(item):
            assert slc.start < self.shape[idx], f'Index of {slc.start} at dim {idx} is out of range'
            assert slc.stop <= self.shape[idx], f'Index of {slc.stop} at dim {idx} is out of range'

        print(item)

        # print(f'Out slice: {item}')
        ## ITEM is now the corrected slice for the virtual array

        ## Actually get data from hicam file and shape into the new array shape
        canvas = np.zeros(
            shape=(
                item[0].stop - item[0].start,
                item[1].stop - item[1].start,
                item[2].stop - item[2].start,
                item[3].stop - item[3].start
            ),
            dtype=self.array.dtype
        )
        # print(canvas.shape)

        ## Extract each color channel
        color_slice = item[0]
        color_slice = [x for x in range(color_slice.start, color_slice.stop,
                                        color_slice.step if color_slice.step is not None else 1)]
        print(f'{color_slice=}')
        # for clr_idx in range(canvas.shape[0]):
        orig = self.array[item[1]]
        while orig.ndim < 3:
            'Ensure all 3 dims are present'
            np.expand_dims(orig, 0)
        print(f'Calculating Lighting correction')
        orig = self.lighting_correction(orig)
        for clr_idx, clr in enumerate(color_slice):
            # clr = clr_idx + item[0].start
            # print(f'CLR index = {clr}')
            if clr == 0:
                color_slice = (
                    slice(0, self.shape[2]),
                    slice(0, self.shape[3])
                )
            if clr == 1:
                color_slice = (
                    slice(0, self.shape[2]),
                    slice(self.shape[3], self.shape[3] * 2)
                )
            if clr == 2:
                color_slice = (
                    slice(self.shape[2], self.shape[2] * 2),
                    slice(0, self.shape[3])
                )
            if clr == 3:
                color_slice = (
                    slice(self.shape[2], self.shape[2] * 2),
                    slice(self.shape[3], self.shape[3] * 2)
                )

            out = orig[:, color_slice[0], color_slice[1]]
            print(f'{out.shape}')
            if self.hicam_color_shifts:
                # out[:] = np.roll(out, self.hicam_color_shifts[str(clr)][0],axis=2)
                # out[:] = np.roll(out, self.hicam_color_shifts[str(clr)][1], axis=1)
                out[:] = np.roll(out, self.hicam_color_shifts[str(clr)][0], axis=1)
                out[:] = np.roll(out, self.hicam_color_shifts[str(clr)][1], axis=2)
            # print(out.shape)
            # print(out)
            out = out[:, item[-2], item[-1]]
            # print(out.shape)
            canvas[clr_idx] = out
            # print(canvas.shape)

        return canvas.squeeze()


####  Image alignments methods
# pip install numpy scikit-image matplotlib SimpleITK

# import SimpleITK as sitk
from skimage import io, img_as_float32, img_as_float, img_as_uint
import numpy as np
import time


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


def sitk_align_translation(fixed, moving, output_offsets=False):
    '''
    Input:
        fixed: numpy array (same shape as moving)
        moving: numpy array (same shape as fixed)

    Output:
        If output_offsets == False (default), an aligned image is returned
        If output_offsets == True, a tuple of pixel offsets is returned

        Images are returned in the same dtype as input
    '''

    dtype = moving.dtype
    # Convert numpy arrays to sitk images
    if fixed.dtype != float:
        fixed = img_as_float32(fixed)
    if moving.dtype != float:
        moving = img_as_float32(moving)
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)
    fixed.SetOrigin((0, 0))
    moving.SetOrigin((0, 0))

    # Calculate alignment
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    # R.SetMetricAsMattesMutualInformation(50)
    # R.SetMetricAsJointHistogramMutualInformation()
    # R.SetMetricAsANTSNeighborhoodCorrelation(2)
    # R.SetMetricAsMeanSquares()
    # R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    # R.SetInterpolator(sitk.sitkNearestNeighbor)
    R.SetInterpolator(sitk.sitkLinear)

    # Pyramidal registration
    R.SetShrinkFactorsPerLevel([6, 2, 1])
    R.SetSmoothingSigmasPerLevel([6, 2, 1])

    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)
    # return outTx
    if output_offsets:
        # Return revered tuple of pixel offsets (sitk and numpy axes are reversed)
        # Returned axes are in order (y,x)
        offsets = outTx.GetParameters()[::-1]
        # print(offsets)
        # invert offsets
        return tuple([-x for x in offsets])

    # Produce aligned image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    R.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)

    # Return numpy array of aligned image
    out = sitk.GetArrayFromImage(out)
    if dtype == out.dtype:
        return out
    if dtype == np.dtype('uint16'):
        return img_as_uint(out)
    if dtype == np.dtype('float32'):
        return img_as_float32(out)
    if dtype == float:
        return img_as_float(out)


def calculate_channels_shift(multi_channel_z_stack, reference_channel=None):
    from skimage.registration import phase_cross_correlation

    if reference_channel is None:
        reference_channel = 0

    shift_dict = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    for ch_idx in range(multi_channel_z_stack.shape[0]):

        ref_array = multi_channel_z_stack[reference_channel]
        reference = ref_array[0]
        if ch_idx != reference_channel:

            moving_array = multi_channel_z_stack[ch_idx]
            moving = moving_array[0]
            for z_idx in range(moving_array.shape[0]):
                # print(z_idx)
                if z_idx % 10 == 0:
                    print(f'Getting Reference and Moving images')
                    reference[:] = ref_array[z_idx]
                    moving[:] = moving_array[z_idx]
                    print(f'Aligning Image {z_idx} of {multi_channel_z_stack.shape[1]}')
                    # shift, error, phasediff = phase_cross_correlation(reference,moving)
                    # print(f'Shift = {shift}, error = {error}, phasediff = {phasediff}')
                    print(f'{reference.shape}, {moving.shape}')
                    shift = sitk_align_translation(reference, moving, output_offsets=True)
                    print(f'{z_idx} of {moving_array.shape[0]}; Shift = {shift}')
                    shift_dict[ch_idx].append(shift)

    return shift_dict


# def calculate_channel_shifts(zarr_spool_location, anchor_channel=0, z_range=None):
#     '''
#         This function takes a 4 channels image and aligns the channels relative to one of the 4 channel (anchor_channel)
#         then calculates the translational shift required to overlay them.
#
#         This version of the function uses a median to calculate the shifts
#         '''
#     # import statistics
#     # np.median = statistics.geometric_mean
#
#     a = open_hicam_array_as_aligned_color_dataset(zarr_spool_location)
#
#     if z_range is None:
#         shift_dict = calculate_channels_shift(a, reference_channel=anchor_channel)
#     elif isinstance(z_range, tuple):
#         image = a[:,z_range[0]:z_range[1]]
#         shift_dict = calculate_channels_shift(image, reference_channel=anchor_channel)
#
#
#     shift_medians = {
#             0:None,
#             1:None,
#             2:None,
#             3:None
#         }
#
#     for key in shift_dict:
#         x = np.median(
#             np.array(
#             [x[0] for x in shift_dict[key]]
#         )
#         )
#         x = 0 if x is np.nan else x
#         y = np.median(
#             np.array(
#             [x[1] for x in shift_dict[key]]
#         )
#         )
#         y = 0 if y is np.nan else y
#
#         try:
#             shift_medians[key] = (round(y),round(x))
#         except Exception:
#             shift_medians[key] = (0, 0)
#     return shift_medians


def calculate_channel_shifts(zarr_spool_location, anchor_channel=0, z_range=None):
    '''
    This function takes a 4 channels image and aligns the channels relative to one of the 4 channel (anchor_channel)
    then calculates the translational shift required to overlay them.

    This version of the function uses a geometric mean to calculate the shifts
    '''
    import statistics

    if isinstance(zarr_spool_location, open_hicam_array_as_aligned_color_dataset):
        a = zarr_spool_location
    elif isinstance(zarr_spool_location, str):
        a = open_hicam_array_as_aligned_color_dataset(zarr_spool_location)

    if z_range is None:
        shift_dict = calculate_channels_shift(a, reference_channel=anchor_channel)
    elif isinstance(z_range, tuple):
        image = a[:, z_range[0]:z_range[1]]
        shift_dict = calculate_channels_shift(image, reference_channel=anchor_channel)

    shift_medians = {
        0: None,
        1: None,
        2: None,
        3: None
    }

    for key in shift_dict:
        print(f'{shift_dict[key]}')
        x = np.array(
            [x[0] for x in shift_dict[key]]
        )
        print(f'{x=}')
        tmp = 0
        if x.size == 0:
            x = np.array([0])
        if int(min(x)) <= 0:
            tmp = abs(min(x)) + 1
            x += tmp
        x = statistics.geometric_mean(
            x
        )
        x -= tmp
        # x = int(x)
        # x = 0 if x is np.nan else x

        y = np.array(
            [x[1] for x in shift_dict[key]]
        )
        tmp = 0
        if y.size == 0:
            y = np.array([0])
        if int(min(y)) <= 0:
            tmp = abs(min(y)) + 1
            y += tmp
        y = statistics.geometric_mean(
            y
        )
        y -= tmp
        # y = int(y)

        # y = 0 if y is np.nan else y

        try:
            shift_medians[key] = (round(y), round(x))
        except Exception:
            shift_medians[key] = (0, 0)
    return shift_medians


#######################################################################################
# Below is for testing and runs when the script is called directly
#######################################################################################

if __name__ == "__main__":
    import os, time

    file = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab6/2025_08_22_HOLiS_NPBB328_Cortex_Slab06/NPBB328-Cortex-Slab06-run999-z13-y044-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli.zst"
    zarr_location = "/bil/users/awatson/test_hicam_out"
    # send_hicam_to_zarr_par_read_once(file, zarr_location, compressor_type='zstd', compressor_level=5, shuffle=1,
    #                                  chunk_depth=128, chunk_lat=128, frames_at_once=1024)

    send_hicam_to_zarr(file, zarr_location, compressor_type='zstd', compressor_level=5, shuffle=1,
                                         chunk_depth=128, frames_at_once=1024)

    #
    # base = '/CBI_FastStore/hillman/test_run_small'
    #
    # spool_files = (
    #     'longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli',
    #     'longRun4-y1-z1_HiCAM FLUO_1875-ST-088.fli',
    #     'longRun3-y1-z0_HiCAM FLUO_1875-ST-272.fli',
    #     'longRun4-y1-z1_HiCAM FLUO_1875-ST-272.fli'
    # )
    #
    # spool_files = [os.path.join(base,x) for x in spool_files]
    #
    # # zarr_locations = [os.path.join(r'/bil/users/awatson/holis',os.path.split(x)[-1] + '_ZARR_OUT') for x in spool_files]
    # zarr_locations = [x + '_ZARR_OUT' for x in spool_files]
    #
    # # def run():
    # #     start = time.time()
    # #     for spool_file, zarr_location in zip(spool_files, zarr_locations):
    # #         send_hicam_to_zarr_par_read_once(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=128)
    # #     stop = time.time()
    # #     print(f'Total time = {stop}')
    #
    #
    # # run()
    # a = [open_hicam_array_as_aligned_color_dataset(zarr_locations[0])]
    # # a = [open_hicam_array_as_aligned_color_dataset(x) for x in zarr_locations]

# header_info, _ = read_header(file2)
# start_index = get_header_size(file2, header_info=header_info)
# with FliOpen(file2) as f:
#     f.seek(start_index)
#     for idx, ii in enumerate(range(40)):
#         a = f.read(1966080*1024)
#         print(f'{idx} len {len(a)}')