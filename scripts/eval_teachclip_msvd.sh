CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12225 \
    eval.py --num_thread_reader=8  --batch_size_val=40 \
    --init_model ["checkpoint path of trained student model"] \
    --data_path ["datasets/msvd_data"] \
    --features_path ["msvd videos path"] \
    --output_dir ["save path of log and checkpoints"] \
    --max_words 32 --max_frames 12 \
    --datatype msvd \
    --feature_framerate 1 --freeze_layer_num 0 \
    --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ["ViT-B/32", or "ViT-B/16"]