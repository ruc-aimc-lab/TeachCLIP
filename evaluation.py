import torch
import sys
from utils.bigfile import BigFile
from tqdm import tqdm
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import argparse
import numpy as np

def get_args(description='CLIP4Clip Distill on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--local_rank', type=int, default=0, help='gpu id')
    parser.add_argument('--text_feat_path', type=str, default='', help='text_feat_path')
    parser.add_argument('--video_feat_path', type=str, default='', help='video_feat_path')
    parser.add_argument('--gt_file_path', type=str, default='', help='gt_file_path')
    args = parser.parse_args()
    return args

def cal_sim_matrix(args):
    query_ids = []
    video_ids = []
    with open(args.gt_file_path, 'r') as f:
        for line in f.readlines():
            query_id, video_id = line.strip().split('\t')
            if query_id not in query_ids:
                query_ids.append(query_id)
            if video_id not in video_ids:
                video_ids.append(video_id)

    video_file = BigFile(args.video_feat_path)
    text_file = BigFile(args.text_feat_path)

    device = torch.device('cuda:{}'.format(args.local_rank))
    text_feats = []
    for query_id in query_ids:
        text_feat = text_file.read_one(query_id)
        text_feats.append(text_feat)
    text_feats = torch.tensor(text_feats).to(device)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    video_feats = []
    for video_id in video_ids:
        video_feat = video_file.read_one(video_id)
        video_feats.append(video_feat)
    video_feats = torch.tensor(video_feats).to(device)
    video_feats = video_feats / video_feats.norm(dim=-1, keepdim=True)

    sim_matrix = torch.einsum('md,nd->mn', text_feats, video_feats)
    sim_matrix_npy = sim_matrix.cpu().numpy()
    return sim_matrix_npy

def get_metrics(sim_matrix, args):
    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        print("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = []
        tmp_video_ids = []
        i = 0
        for line in open(args.gt_file_path, 'r'):
            query_id, video_id = line.strip().split('\t')
            if video_id not in tmp_video_ids:
                tmp_video_ids.append(video_id)
                cut_off_points2len_.append(i)
            i = i + 1
        cut_off_points2len_ = cut_off_points2len_[1:]
        cut_off_points2len_.append(i)

        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        print("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        print("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    print("Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    print("Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

def main():
    args = get_args()
    sim_matrix = cal_sim_matrix(args)
    get_metrics(sim_matrix, args)

if __name__ == "__main__":
    main()
