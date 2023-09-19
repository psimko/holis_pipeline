# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:25:09 2022

@author: awatson
"""

import os
import re
import sys
import glob
import math
import shutil
import numpy as np
from natsort import natsorted
from skimage import io, img_as_float32, img_as_uint
from dask.delayed import delayed
import tifffile
from distributed import Client


def run():
    """
    Takes two command-line arguments:
    - path to tiff_stacks folder (e.g. /bil/proj/rf1hillman/2023_08_15_combinatorialSlides2_AI7_EH5f/withoutOverlap/AI7_EH5f3/tiff_stacks)
    - path to output folder, where composite tiffs will reside (e.g. /bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/AI7_EH5f3/composites)

    run: python fuse_tiffs.py <path_to_tiff_stacks> <path_to_output_folder>
    """
    cpus = os.cpu_count()
    workers = cpus//2
    threads = 2
    print(f"Using {workers} workers")
    with Client(n_workers=workers,threads_per_worker=threads) as client:
        
        def shift_image(image,y,x):
            if x==0 and y==0:
                return image
            
            print('Shifting image y {}, x {}'.format(y,x))
            canvas = np.zeros_like(image)
            if y >= 0 and x >= 0:
                canvas[y:,x:] = image[:-y,:-x]
                
            
            elif y < 0 and x >= 0:
                canvas[:-abs(y),x:] = image[abs(y):,:-x]
            
            elif y >= 0 and x < 0:
                canvas[y:,:-abs(x)] = image[:-y,abs(x):]
                
            elif y < 0 and x < 0:
                canvas[:-abs(y),:-abs(x)] = image[abs(y):,abs(x):]
            
            return canvas

        def flat_field(raw,corr,mult_factor):
            '''
            Takes single line corections and expands to correct
            full image.  Assumes correction is float in uint values
            '''
            # raw -= 90 #Fudge factor to compensate for no dark field
            # raw[raw<0] = 0
            out = raw/corr[:,np.newaxis]
            out *= mult_factor
            out = out.astype('uint16')
            return out
        
        def flat_field_z(raw,corr,mult_factor):
            '''
            Takes single line corections and expands to correct
            full image.  Assumes correction is float in uint values
            '''
            # raw -= 90 #Fudge factor to compensate for no dark field
            # raw[raw<0] = 0
            out = raw / corr
            out *= mult_factor
            out = out.astype('uint16')
            return out
        
        def write_image(path,array,tile=(1024,1024)):
            print('Writing {}'.format(path))
            tifffile.imwrite(path,array,tile=tile)
            # tifffile.imwrite(path,array,tile=tile, compression='lzw')
            return True

        tiffFolderLocation = sys.argv[1]
        out_folder = sys.argv[2]
        print("Out folder", out_folder)
        import time
        # time.sleep(3)
        os.makedirs(out_folder,exist_ok=True)
        
        lighting = False
        if lighting:
            flats_location = r'Z:\Alan\hillman\SR3B6_wholeBrain_flatFields'
            flats_location = r'/CBI_FastStore/Alan/hillman/SR3B6_wholeBrain_flatFields'
            
            z_lighting = r'/CBI_FastStore/Alan/hillman/SR3B6_wholeBrain_flatZ'
            z_lighting = sorted(glob.glob(z_lighting + '/*.tif'))
            z_lighting = [io.imread(x) for x in z_lighting]
            z_lighting_mean = [x.mean() for x in z_lighting]
            
            # List flatfield files
            flats = glob.glob(flats_location + '/*.tif')
            # Read all flatfield files
            flats = [io.imread(x) for x in flats]
            flats_mean = [x.mean() for x in flats]
        
        # List all tiff files recursively
        tiff_files = glob.glob(tiffFolderLocation + '/**/*.tiff',recursive=True)
        print("Total files:", len(tiff_files))
        

        # max_z = [a[a.find('plane')+5:a.find('plane')+8] for a in tiff_files]
        max_z = [re.findall(r"_plane\d{3}", a)[0] for a in tiff_files]
        z_nums = [int(a.replace("_plane", "")) for a in max_z]
        max_z = max(z_nums)
        min_z = min(z_nums)
        print("Max plane", max_z)
        print("Min plane", min_z)

        # max_c = [a[a.find('_c')+2:a.find('_c')+3] for a in tiff_files]
        try:
            max_c = [re.findall(r"_ch\d", a)[0] for a in tiff_files]
            channel_prefix = "ch"
        except IndexError:
            max_c = [re.findall(r"_c\d", a)[0] for a in tiff_files]
            channel_prefix = "c"
        c_nums = [int(a.replace(f"_{channel_prefix}", "")) for a in max_c]
        max_c = max(c_nums)
        min_c = min(c_nums)
        print("Max c", max_c)
        print("Min c", min_c)
        
        # max_slabs = [a[a.find('_z')+2:a.find('_z')+4] for a in tiff_files]
        max_slabs = [re.findall(r"_z\d{2}", a)[0] for a in tiff_files]
        slab_nums = [int(a.replace("_z", "")) for a in max_slabs]
        max_slabs = max(slab_nums)
        min_slabs = min(slab_nums)
        print("Max slab", max_slabs)
        print("Min slab", min_slabs)

        # Trim xpixels off of each side of each strip
        x_trim = 0 #54//2
        
        # Trim zlayers from each sub stack
        z_trim = 0 #198//2

        x_shift = y_shift = 0
        
        num = 0
        run = 0
        z_planes = []
        failed = []
        skip = False
        slab_idx = 0
        to_compute = []
        for z_ittr, z in enumerate(range(min_slabs, max_slabs+1)):
            test = '_z' + str(z).zfill(2)
            layer = [x for x in tiff_files if test in x]
            print("Files in z layer", len(layer))
            slab_idx += 1
            for ch in range(min_c, max_c+1):
                color = f'_{channel_prefix}' + str(ch)
                channel = [a for a in layer if color in a]
                print("Files in channel", len(channel))
                z_idx = 0
                bottom_skip_idx = 0
                p = -1
                
                for z_loop_idx,zplane in enumerate(range(min_z,max_z+1)):
                    
                    while True:
                        p += 1
                        try:
                            btm = max_z - z_trim
                            if p < z_trim or p > btm:
                                # print('Skipping z-layer {}'.format(p))
                                if p > btm:
                                    bottom_skip_idx += 1
                                if bottom_skip_idx >= 50:
                                    break
                                continue
                            test = '_plane{}'.format(str(p).zfill(3))
                            plane = [x for x in channel if test in x]
                            print("Files in plane", len(plane))
                            if plane == []:
                                # print('None left')
                                continue
                            files = natsorted(plane)
                            
                            # z_plane = files[0].split('_plane')[-1].split('_')[0]
                            # z_plane = int(z_plane)
                            z_idx += 1
                            z_plane = z_idx
                            if slab_idx>1:
                                z_plane = (max(z_planes)*(slab_idx-1)) + z_plane
                            else:
                                z_planes.append(z_plane)
                            z_plane = str(z_plane).zfill(4)
                            image_name = 'composite_c{}_z{}.tif'.format(ch,z_plane)
                            
                            # print('Reading Images')
                            plane = [delayed(io.imread)(x) for x in files]
                            if x_trim > 0:
                                plane = [x[x_trim:-x_trim] for x in plane]
                            if lighting:
                                # print('Forming Flatfield Correction')
                                if x_trim > 0:
                                    plane = [delayed(flat_field)(x,flats[ch-1][x_trim:-x_trim][:,z_loop_idx],flats_mean[ch-1]) for x in plane]
                                else:
                                    plane = [delayed(flat_field)(x,flats[ch-1][:,z_loop_idx],flats_mean[ch-1]) for x in plane]
                                
                                plane = [delayed(flat_field_z)(x,z_lighting[ch-1][z_loop_idx],z_lighting_mean[ch-1]) for x in plane]
                                
                            plane = delayed(np.concatenate)(plane,0)
                            # plane = dask.compute(plane)[0]
                            
                            if (x_shift != 0 or y_shift != 0) and slab_idx > 1:
                                plane = delayed(shift_image)(plane,y_shift*z_ittr,x_shift*z_ittr)
                                # if slab_idx > 5: #HACK for missing z08
                                #     plane = delayed(shift_image)(plane,y_shift*(z_ittr+1),x_shift*(z_ittr+1))
                                # else:
                                #     plane = delayed(shift_image)(plane,y_shift*z_ittr,x_shift*z_ittr)
                            if num == 0:
                                avg = delayed(np.zeros)(plane.shape,dtype=np.dtype('float32'))
                                last_image = plane
                            else:
                                last_image = plane
                            # avg += img_as_float32(plane)
                            
                                
                            num += 1
                            print('Queueing slab {} of {} : {}'.format(z,max_slabs,image_name))
                            # image_name = 'composite_c{}_z{}.tif'.format('{}{}'.format(c,ch),str(z_planes+p).zfill(3))
                            # io.imsave(os.path.join(out_folder,image_name),plane)
                            write = delayed(write_image)(os.path.join(out_folder,image_name),plane,tile=(512,512))
                            to_compute.append(write)
                        except Exception as e:
                            print(e)
                            print('Plane Failed, Saving last image instead')
                            failed.append(image_name)
                            tifffile.imwrite(os.path.join(out_folder,image_name),last_image,tile=(512,512))
        
        print('Computing')
        complete = []
        import time
        while len(to_compute) > 0:
            complete.append(client.compute(to_compute.pop()))
            while len(complete) > round(cpus*1.25):
                time.sleep(1)
                complete = [x for x in complete if x.status != 'finished']
                
            
        plane = client.gather(complete)

    organize_composites(out_folder, channel_prefix, min_c, max_c)


def fuse_in_x(in_dir, out_dir):
    """
    Only needed if strips are split in multiple x chunks
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cpus = os.cpu_count()
    workers = cpus//2
    threads = 2
    with Client(n_workers=workers,threads_per_worker=threads) as client:
        zmax = 1
        ymax = 2
        cmax = 1
        # planes = 441
        planes = 701
        to_compute = []
        for z in range(1, zmax+1):
            for y in range(1, ymax+1):
                print("Processing y", y)
                yz_folder = os.path.join(out_dir, f"mouseBrain_dualCam_z{str(z).zfill(2)}_y{str(y).zfill(2)}")
                if not os.path.exists(yz_folder):
                    os.makedirs(yz_folder)
                for c in range(1, cmax+1):
                    print("Processing c", c)
                    folders = sorted(glob.glob(os.path.join(in_dir, f"mouseBrain_dualCam_z{str(z).zfill(2)}_y{str(y).zfill(2)}_x??_ch{c}")))
                    print("Folders with x", len(folders))
                    all_files = []
                    for folder in folders:
                        all_files.extend(glob.glob(os.path.join(folder, "*.tiff")))
                    print("Total files in x folders", len(all_files))
                    for plane in range(1, planes+1):
                        print("plane", plane)
                        if os.path.exists(os.path.join(yz_folder, f"mouseBrain_dualCam_z{str(z).zfill(2)}_y{str(y).zfill(2)}_plane{str(plane).zfill(3)}_c{c}.tiff")):
                            print("skipping")
                            continue
                        files =  [x for x in all_files if x.endswith(f"plane{str(plane).zfill(3)}.tiff")]
                        print(*files, sep="\n")
                        to_concatenate = [delayed(tifffile.imread)(x) for x in all_files if x.endswith(f"plane{str(plane).zfill(3)}.tiff")]
                        print("to concatenate", len(to_concatenate))
                        print(to_concatenate[0].shape)
                        print(to_concatenate[1].shape)
                        print(to_concatenate[2].shape)
                        print(to_concatenate[3].shape)
                        concatenated = delayed(np.concatenate)(to_concatenate, axis=1)
                        print("Concatenated", concatenated.shape)
                        saved = delayed(tifffile.imwrite)(
                            os.path.join(yz_folder, f"mouseBrain_dualCam_z{str(z).zfill(2)}_y{str(y).zfill(2)}_plane{str(plane).zfill(3)}_c{c}.tiff"),
                            concatenated,
                            tile=(512,512)
                            # compression='lzw'
                        )
                        to_compute.append(saved)

        print('Computing')
        complete = []
        import time
        while len(to_compute) > 0:
            complete.append(client.compute(to_compute.pop()))
            while len(complete) > round(cpus * 1.25):
                time.sleep(1)
                complete = [x for x in complete if x.status != 'finished']

        plane = client.gather(complete)


def organize_composites(out_folder, channel_prefix, channel_min, channel_max):
    all_composites = sorted(glob.glob(os.path.join(out_folder, '*.tif')))
    nuclei_folder = os.path.join(out_folder, "nuclei")
    colors_folder = os.path.join(out_folder, "colors")
    for channel in range(channel_min, channel_max + 1):
        channel_composites = [x for x in all_composites if f"{channel_prefix}{channel}" in x]
        print(f"Number of channel {channel} composites: {len(channel_composites)}")
        if channel == 1:
            dest = os.path.join(nuclei_folder, str(channel))
            if not os.path.exists(dest):
                os.makedirs(dest)
            res = [shutil.move(f, dest) for f in channel_composites]
        else:
            dest = os.path.join(colors_folder, str(channel))
            if not os.path.exists(dest):
                os.makedirs(dest)
            res = [shutil.move(f, dest) for f in channel_composites]


if __name__ == '__main__':
    run()
