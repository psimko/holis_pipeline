from holis_pipeline.preprocessing_functions import infer_z, infer_y
import os, sys
from glob import glob


input_dir = sys.argv[1] 
y_max = 10
z_max = 13
file_pattern = '*_transformed*'

def main():
    file_glob = glob(os.path.join(input_dir, file_pattern)) 
    print(len(file_glob))
    converted = set()
    for file in file_glob:
        y = int(infer_y(file))
        z = int(infer_z(file))
        converted.add((y,z))
    
    missing = []
    for z in range(1,z_max+1):
        for y in range(1,y_max+1):
            if (y,z) not in converted:
                missing.append((y,z))

    for (y,z) in missing:
        print(f'Missing file y={y}, z={z}')

if __name__ == "__main__":
    main()