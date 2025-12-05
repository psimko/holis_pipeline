import argparse, os, re
from pathlib import Path
import numpy as np
from tifffile import imwrite
from holis_pipeline.utils.zarr_related import read_omehans 


Y_KEY_RE = re.compile(r'y(\d+)', re.IGNORECASE)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", type=int)                      # 0 nuclei, else splitter
    ap.add_argument("processing")                             # e.g. None/bg_subtracted/laser_corrected/registered/unmixed
    ap.add_argument("path_ome", type=Path)                    # path to .../<something>/omehans
    ap.add_argument("tiffs_root", type=Path)                  # root where we store tiffs
    args = ap.parse_args()

    path_ome: Path = args.path_ome
    assert path_ome.name == "omehans", f"path_ome must end with 'omehans', got {path_ome}"

    # Read dask array and compute
    print(f"[worker] Reading {path_ome}")
    image_dask = read_omehans(path_ome)         # expected shape like (37000, 1024, 1280)
    print("Shape of the strip:", image_dask.shape)
    image = image_dask.compute()
    print("Computed into memory.")

    # Build output dir mirroring the parent of 'omehans'
    # e.g. .../<...>/<run-yNNN-...>/{stripe_0000.tif,...}
    rel_name = path_ome.parent.name
    out_dir = args.tiffs_root / rel_name
    _ensure_dir(out_dir)

    # Save per-z as TIF
    # image is (Z?, Y, X) or (T?, Y, X) — original loop used image.shape[1] as Z;
    # the code extracted plane = image[:, z, :] -> so axis-1 is "z", axis-0 is "rows"
    # We keep that to stay faithful.
    depth = image.shape[1]
    for z in range(depth):
        out_path = out_dir / f"stripe_{z:04d}.tif"
        if out_path.exists():
            continue
        if z % 100 == 0:
            print("saving", z)
        plane = image[:, z, :]
        imwrite(out_path.as_posix(), plane.astype(image.dtype))

    print(f"[worker] Done: {out_dir}")

if __name__ == "__main__":
    main()