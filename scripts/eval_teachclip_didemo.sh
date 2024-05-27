CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12229 \
    eval.py --num_thread_reader=8 \
    --init_model ["checkpoint path of trained student model"] \
    --data_path ["datasets/didemo_data"]\
    --features_path ["didemo videos path"] \
    --output_dir ["save path of log and checkpoints"] \
    --max_words 64 --max_frames 64 --batch_size_val=20 \
    --datatype didemo \
    --feature_framerate 1 --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ["ViT-B/32", or "ViT-B/16"]
