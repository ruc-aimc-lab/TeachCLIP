test_collection=$1
videoset=$2
model_name=$3 # $train_collection/Models/$config/run$ID

train_collection=$(echo "$model_name" | cut -d'/' -f1)
config=$(echo "$model_name" | cut -d'/' -f3)
run=$(echo "$model_name" | cut -d'/' -f4)

python -m torch.distributed.launch extract_feat.py --datatype video \
    --local_rank=3 --num_thread_reader=8 --batch_size=100 \
    --videofile_path "data/datasets/$test_collection/VideoSet/$videoset.txt" \
    --video_path "data/datasets/$test_collection/VideoData" \
    --output_dir "data/datasets/$test_collection/FeatureData/$videoset/$train_collection/$config/$run" \
    --init_model "data/datasets/$model_name/best_model.bin" \
    --max_frames=12 --max_words=32 --feature_framerate 1 --slice_framepos 2 \
    --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32

