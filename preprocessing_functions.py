#######################################################################################
#########   Preprocessing functions for hicam (.fli) files - Holis project ############ 
#######################################################################################

import numpy as np
from ast import literal_eval
import json
from pprint import pprint as print
import os
from numcodecs import Blosc
# from numcodecs import blosc
# blosc.set_nthreads(16)
from zarr_stores.h5_nested_store import H5_Nested_Store
import zarr
from skimage import io
import dask
from dask import delayed

#######################################################################################

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

#######################################################################################

def send_hicam_to_zarr_par_read_once(hicam_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, chunk_lat=128, frames_at_once=1024):
    """
    Function for converting a hicam .fli file into an omezarr file.

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


