from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import time
import argparse
import sys
from tqdm import tqdm

from dataloaders.data_dataloaders import DATALOADER_DICT
from modules.modeling import CLIP4Clip
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.tokenization_clip import Tokenizer

import utils.txt2bin as txt2bin

def get_args(description='TeachCLIP Feature Extraction on a sigle GPU'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datatype", default="video", type=str, required=True, choices=['video', 'text'], help="Point the dataset to extract feature.")
    
    # arguments for dataloder
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0, help='gpu id')
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for all dataloder')

    parser.add_argument('--overwrite', action='store_true', help='overwrite output feature file if true')
    parser.add_argument('--queryfile_path', type=str, default='data/datasets/msrvtt/QuerySet/msrvtt1k-test-query.txt', help='query id, query')
    parser.add_argument('--videofile_path', type=str, default='data/datasets/msrvtt/VideoSet/msrvtt1k-test.txt', help='video id')
    parser.add_argument('--video_path', type=str, default='data/datasets/msrvtt/VideoData', help='video data dir')
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the feature file will be written.")

    # arguments for model
    parser.add_argument("--init_model", default=None, type=str, required=True, help="Initial model.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")
    parser.add_argument('--sim_header', type=str, default="seqTransf",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")

    # arguments for video feature extraction
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--image_resolution', type=int, default=224, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")

    args = parser.parse_args()

    return args

def set_seed(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def init_model(args, device):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.to(device)

    return model

def extract_video_feature(args, model, video_dataloader, device):
    model.eval()

    id_feature_path = os.path.join(args.output_dir, 'id.feature.txt')
    if os.path.exists(id_feature_path):
    	if args.overwrite:
        	print('%s exists. overwrite', id_feature_path)
    	else:
    		print('%s exists. skip', id_feature_path)
    		sys.exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    open_type = 'w'
    fw = open(id_feature_path, open_type)

    with torch.no_grad():
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        print("***** Extracting video featrures *****")
        for batch in tqdm(video_dataloader):
            video_ids = batch[0]
            batch = tuple(t.to(device) for t in batch[1:])
            video, video_mask = batch

            # video_features: [batch_size, out_dim]
            video_features = model.get_video_output(video, video_mask)

            # write video features to txt file
            video_features_numpy = video_features.cpu().numpy()
            for i in range(len(video_ids)):
                line = str(video_ids[i]) + ' ' + ' '.join([str(num) for num in video_features_numpy[i, :]]) + '\n'
                fw.write(line)
    
    fw.close()
    # transform to bin format
    print("***** txt to bin format *****")
    overwrite = args.overwrite
    txt2bin.process(0, [id_feature_path], args.output_dir, overwrite)
    # delete id.feature.txt
    os.remove(id_feature_path)

def extract_text_feature(args, model, text_dataloader, device):
    model.eval()

    id_feature_path = os.path.join(args.output_dir, 'id.feature.txt')
    if os.path.exists(id_feature_path):
    	if args.overwrite:
        	print('%s exists. overwrite', id_feature_path)
    	else:
    		print('%s exists. skip', id_feature_path)
    		sys.exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    open_type = 'w'
    fw = open(id_feature_path, open_type)

    with torch.no_grad():
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        print("***** Extracting text featrures *****")
        tokenizer = Tokenizer(max_words=args.max_words)
        for batch in tqdm(text_dataloader):
            sentence_ids, sentences = batch[0], batch[1]
            
            # text tokenization
            input_ids, input_mask, segment_ids = tokenizer._get_text(sentence_ids, sentences)
            input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)

            # text_features: [batch_size, out_dim]
            text_features = model.get_sequence_output(input_ids, segment_ids, input_mask).squeeze()
            
            # write text features to txt file
            text_features_numpy = text_features.cpu().numpy()
            if torch.is_tensor(sentence_ids):
                sentence_ids = sentence_ids.cpu().numpy()
            for i in range(len(sentence_ids)):
                line = str(sentence_ids[i]) + ' ' + ' '.join([str(num) for num in text_features_numpy[i, :]]) + '\n'
                fw.write(line)
    
    fw.close()
    # transform to bin format
    print("***** txt to bin format *****")
    txt2bin.process(0, [id_feature_path], args.output_dir, args.overwrite)
    # delete id.feature.txt
    os.remove(id_feature_path)

def main():
    args = get_args()
    set_seed(args)
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.local_rank))
    else:
        device = torch.device('cpu')
        raise Error('GPU is not available, infer on cpu is too slow!')

    ## ####################################
    # model loading
    ## ####################################
    model = init_model(args, device)

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT
    video_dataloader = None
    if args.datatype == 'video':
        video_dataloader, video_set_length = DATALOADER_DICT[args.datatype](args)
        print("***** Video feature extraction *****")
        print("  Num examples = ", video_set_length)
        print("  Batch size = ", args.batch_size)
        print("  Num steps = ", len(video_dataloader))
    elif args.datatype == 'text':
        text_dataloader, text_set_length = DATALOADER_DICT[args.datatype](args)
        print("***** Text feature extraction *****")
        print("  Num examples = ", text_set_length)
        print("  Batch size = ", args.batch_size)
        print("  Num steps = ", len(text_dataloader))
    
    ## ####################################
    # featue extraction
    ## ####################################
    if args.datatype == 'video':
        extract_video_feature(args, model, video_dataloader, device)
    elif args.datatype == 'text':
        extract_text_feature(args, model, text_dataloader, device)
    else:
        print('not implemented!')

if __name__ == "__main__":
    main()