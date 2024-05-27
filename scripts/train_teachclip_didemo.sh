CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=12229 \
    train.py --teacher_num 1 \
    --init_teacher1_model ["checkpoint path of pretrained teacher model"] \
    --teacher1_name ["XCLIP", or "TS2Net", or "XPool"] \
    --do_train --num_thread_reader=8 \
    --lr 1e-4 --batch_size=64 --batch_size_val=20 --epochs=10 --n_display=10 \
    --data_path ["datasets/didemo_data"]\
    --features_path ["didemo videos path"] \
    --output_dir ["save path of log and checkpoints"] \
    --max_words 64 --max_frames 64 \
    --datatype didemo \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32
