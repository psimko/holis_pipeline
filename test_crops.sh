STEP=1
NTESTS=5
SOURCE="/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_tiffs/transformed_noCrop/"
OUTDIR="/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/crop_tests/"
mkdir -p "$OUTDIR"

for ((i=0; i<NTESTS; i++)); do
    #/ZTopCROP=$((0+i*STEP))
    #/ZBottomCROP=$((0+i*STEP))
    ZTopCROP=30
    ZBottomCROP=30
    #/YRightCROP=41
    #/YLeftCROP=41
    YRightCROP=$((41+i*STEP))
    YLeftCROP=$((41+i*STEP))

    python tiffVols_to_composites.py \
        "$SOURCE" \
        --output_tif "$OUTDIR/composite_colors_shifted_originalCropMinusYadded_ZTopcrop${ZTopCROP}_ZBottomcrop${ZBottomCROP}_YRightcrop${YRightCROP}_YLeftcrop${YLeftCROP}.tif" \
        --kind colors \
        --folder_glob "*-088.fli*_transformed*" \
        --no_norm \
        --channel 0 \
        --crop_bottom_added "$ZBottomCROP" \
        --crop_top_added "$ZTopCROP" \
        --crop_right_added "$YRightCROP" \
        --crop_left_added "$YLeftCROP" 
done

