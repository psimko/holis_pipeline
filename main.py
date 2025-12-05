#main.py
from pathlib import Path
from process_scan import run_pipeline

def main(input_dir: str, out_root: str):
    files = sorted(Path(input_dir).glob("*Exc*272.fli.zst"))
    print(f"Found {len(files)} inputs.")
    for i, nf in enumerate(files, 1):
        odir = Path(out_root) #/ nf.stem.stem  # e.g. A01_272.fli.zst -> /out/A01_272
        #if odir.exists():
        #    print(f"[{i}/{len(files)}] SKIP (exists): {odir}")
        #    continue
        odir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(files)}] RUN  : {nf} -> {odir}")
        run_pipeline(str(nf), str(odir))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input-dir> <out-root>")
        sys.exit(1)
    input_dir = sys.argv[1]
    out_root  = sys.argv[2]
    main(input_dir, out_root)

    # Note:
    # For Slab 6:
    # input_dir: /bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/
    # ouput_dir /bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed/