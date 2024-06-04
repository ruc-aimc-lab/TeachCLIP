from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
import yaml
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.modeling_xclip import XCLIP
from modules.modeling_ts2net import TS2Net
from modules.modeling_xpool import XPool
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
from modules.until_module import CrossEn,MaxMarginRankingLoss
import torch.nn.functional as F
import time

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CLIP4Clip Distill on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--config_path', type=str, required=True, help='Path to the config.')

    parser.add_argument('--teacher_num', type=int, default=1, help='')
    parser.add_argument('--init_teacher1_model', type=str, default='', help='')
    parser.add_argument('--init_teacher2_model', type=str, default='', help='')
    parser.add_argument('--init_teacher3_model', type=str, default='', help='')
    parser.add_argument('--teacher1_name', type=str, default='XCLIP', help='')
    parser.add_argument('--teacher2_name', type=str, default='', help='')
    parser.add_argument('--teacher3_name', type=str, default='', help='')
    
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in data.items():
            for k, v in value.items():
                setattr(args, k, v)


    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.overwrite:
        args.output_dir = os.path.join(args.output_dir, 'run0')
    else:
        run_id = len(os.listdir(args.output_dir))
        args.output_dir = os.path.join(args.output_dir, 'run{}'.format(run_id))
    time.sleep(3)
    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    time.sleep(3)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, task_config=args)

    model.to(device)
    # print(model.clip)

    return model

def init_teacher_models(args, device, n_gpu, local_rank):
    teacher_models = []
    logger.info("***** Teacher Num:{} *****".format(args.teacher_num))
    for i in range(1,args.teacher_num+1):
        init_model_path = eval("args.init_teacher{}_model".format(i))
        if init_model_path:
            model_state_dict = torch.load(init_model_path, map_location='cpu')
        else:
            model_state_dict = None

        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

        model_name =  eval("args.teacher{}_name".format(i))
        print("model_name",model_name)
        if model_name=="XCLIP":
            model = XCLIP.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
        elif model_name=="CLIP4Clip":
            model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
        elif model_name=="TS2Net":
            model = TS2Net.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
        elif model_name=="XPool":
            model = XPool(task_config=args)
            model.load_state_dict(model_state_dict['state_dict'], strict=False)
        else:
            logger.info("Error about Teacher Name")
            exit()

        model.to(device)
        model.eval()
        teacher_models.append(model)
        # print(model.clip)

    return teacher_models

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    new_para = []

    for n, p in param_optimizer:
        if n=="frameLinear.weight" or n=="frameLinear.bias" or n=="frameLinear2.weight" or n=="frameLinear2.bias":
            new_para.append((n, p))
            param_optimizer.remove((n, p))


    #frameLinear.weight
    # print(list(param_optimizer.keys()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0},
        {'params': [p for n, p in new_para], 'weight_decay': weight_decay, 'lr': args.lr * 10}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def train_epoch(epoch, args, model, teacher_models, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0.0
    
    gt_loss_fun = CrossEn()
    distill_method = args.distill_method #"pear" #"ce",l1
    fine_method = args.fine_method #"ce" #"pear",l1
    distill_loss_fun = torch.nn.SmoothL1Loss(reduction='mean')

    
    for step, batch in enumerate(train_dataloader):
        # if n_gpu == 1:
        #     # multi-gpu does scattering it-self
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        #####################superparam##############
        beta_param = args.beta
        # lambda_param = 1.0
        
        input_ids, input_mask, segment_ids, video, video_mask = batch
        sim_matrix, Frameweight = model(input_ids, segment_ids, input_mask, video, video_mask,return_fine=True)

        sim_matrix = sim_matrix/beta_param
        sim_loss1 = gt_loss_fun(sim_matrix)
        sim_loss2 = gt_loss_fun(sim_matrix.T)
        gt_loss = (sim_loss1 + sim_loss2) / 2

        teacher_sim_matrixs = []
        sentence2frames_sims = []

        with torch.no_grad():
            for i,teacher_model in enumerate(teacher_models):
                if eval("args.teacher{}_name".format(i+1)) == "XCLIP":
                    teacher_sim_matrix,sentence2frames_sim = teacher_model(input_ids, segment_ids, input_mask, video, video_mask,return_fine=True)
                    teacher_sim_matrix = teacher_sim_matrix/beta_param
                    teacher_sim_matrixs.append(teacher_sim_matrix.detach())
                    sentence2frames_sims.append(sentence2frames_sim.detach())
                elif eval("args.teacher{}_name".format(i+1)) == "TS2Net":
                    teacher_sim_matrix, sentence2frames_sim = teacher_model(input_ids, segment_ids, input_mask, video, video_mask,return_fine=True)
                    teacher_sim_matrix = teacher_sim_matrix/beta_param
                    teacher_sim_matrixs.append(teacher_sim_matrix.detach())
                    sentence2frames_sims.append(sentence2frames_sim.detach())
                elif eval("args.teacher{}_name".format(i+1)) == "XPool":
                    teacher_sim_matrix, sentence2frames_sim = teacher_model(input_ids, segment_ids, input_mask, video, video_mask,return_fine=True)
                    teacher_sim_matrix = teacher_sim_matrix/beta_param
                    teacher_sim_matrixs.append(teacher_sim_matrix.detach())
                    sentence2frames_sims.append(sentence2frames_sim.detach())
            
        distill_loss = 0.0
        for teacher_sim_matrix in teacher_sim_matrixs:
            if distill_method=="ce":
                logp = F.log_softmax(sim_matrix, dim=-1)
                logpT = F.log_softmax(sim_matrix.T, dim=-1)
                tmp_distill_loss1 = -1.0*(teacher_sim_matrix.softmax(-1))*logp
                tmp_distill_loss2 = -1.0*(teacher_sim_matrix.T.softmax(-1))*logpT
                tmp_distill_loss = tmp_distill_loss1.mean()+tmp_distill_loss2.mean()
                distill_loss = distill_loss+tmp_distill_loss

            elif distill_method=="pear":
                soft_p = sim_matrix.softmax(-1)
                soft_p_T = sim_matrix.T.softmax(-1)
                soft_q = teacher_sim_matrix.softmax(-1)
                soft_q_T = teacher_sim_matrix.T.softmax(-1)
                cor_soft_p = soft_p-soft_p.mean(-1).view(-1,1)
                cor_soft_p_T = soft_p_T-soft_p_T.mean(-1).view(-1,1)
                cor_soft_q = soft_q-soft_q.mean(-1).view(-1,1)
                cor_soft_q_T = soft_q_T-soft_q_T.mean(-1).view(-1,1)
                tmp_distill_loss1 = 1-((cor_soft_p*cor_soft_q).sum(1)/(cor_soft_p.norm(dim=1)*cor_soft_q.norm(dim=1)))
                tmp_distill_loss2 = 1-((cor_soft_p_T*cor_soft_q_T).sum(1)/(cor_soft_p_T.norm(dim=1)*cor_soft_q_T.norm(dim=1)))
                tmp_distill_loss = tmp_distill_loss1.mean()+tmp_distill_loss2.mean()
                distill_loss = distill_loss+tmp_distill_loss
            elif distill_method=="l1":
                tmp_distill_loss = distill_loss_fun(teacher_sim_matrix,sim_matrix)
                distill_loss = distill_loss+tmp_distill_loss.mean()
            else:
                print("error distill method.")
        distill_loss = distill_loss/len(teacher_sim_matrixs)

        fine_loss = 0.0
        for sentence2frames_sim in sentence2frames_sims:
            # print("F",Frameweight[0:10])
            # print("sim",sentence2frames_sim[0:10])
            if fine_method=="ce":
                tmp_fine_loss = ((-1)*torch.log(Frameweight)*sentence2frames_sim)
                fine_loss = fine_loss+tmp_fine_loss.mean()
            elif fine_method=="pear":
                Frameweight_tmp = Frameweight-Frameweight.mean(-1).view(-1,1)
                sentence2frames_sim_tmp = sentence2frames_sim-sentence2frames_sim.mean(-1).view(-1,1)
                tmp_fine_loss = 1-((Frameweight_tmp*sentence2frames_sim_tmp).sum(1)/(Frameweight_tmp.norm(dim=1)*sentence2frames_sim_tmp.norm(dim=1)))
                fine_loss = fine_loss+tmp_fine_loss.mean()

            elif fine_method=="l1":
                tmp_fine_loss = distill_loss_fun(Frameweight,sentence2frames_sim)
                fine_loss = fine_loss+tmp_fine_loss.mean()
            # cor_Frameweight = Frameweight-Frameweight.mean(-1).view(-1,1)
            # cor_sentence2frames_sim = sentence2frames_sim-sentence2frames_sim.mean(-1).view(-1,1)
            # tmp_fine_loss = 1-((cor_Frameweight*cor_sentence2frames_sim).sum(1)/(cor_Frameweight.norm(dim=1)*cor_sentence2frames_sim.norm(dim=1)))
            # tmp_fine_loss = tmp_fine_loss.mean()
            # fine_loss = fine_loss+tmp_fine_loss
        fine_loss = fine_loss/len(sentence2frames_sims)

        
        start_epoch = 0
        if epoch >= start_epoch:
            if distill_method=="ce":
                # loss =  distill_loss + fine_loss 
                loss = gt_loss + distill_loss * 10 + fine_loss 
            else:
                # loss =  distill_loss + fine_loss #0.2 x 1 x 2 
                loss = gt_loss*0.2 + distill_loss + fine_loss*2.0  #0.2 x 1 x 2 
                # loss = gt_loss + distill_loss*0.2
        else:
            if distill_method=="ce":
                # loss =  distill_loss 
                loss = gt_loss + distill_loss * 10
            else:
                # loss =  distill_loss 
                loss = gt_loss + distill_loss 
                


        
        loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, GTLoss: %f, DistillLoss: %f, FineLoss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            float(gt_loss.mean()),
                            float(distill_loss.mean()),
                            float(fine_loss.mean()),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, _, Frameweight = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type,return_fine=True)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    
    np.save(os.path.join(args.output_dir,"sim_matrix.npy"),sim_matrix)

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    model = init_model(args, device, n_gpu, args.local_rank)

    teacher_models = init_teacher_models(args, device, n_gpu, args.local_rank)


    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.data_type in DATALOADER_DICT

    assert DATALOADER_DICT[args.data_type]["test"] is not None \
           or DATALOADER_DICT[args.data_type]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.data_type]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.data_type]["test"](args, tokenizer)

    if DATALOADER_DICT[args.data_type]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.data_type]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train
    ## ####################################
    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.data_type]["train"](args, tokenizer)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    coef_lr = args.coef_lr
    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

    best_score = 0.00001
    best_output_model_file = "None"
    ## ##############################################################
    # resume optimizer state besides loss to continue train
    ## ##############################################################
    resumed_epoch = 0
    if args.resume_model:
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resumed_epoch = checkpoint['epoch']+1
        resumed_loss = checkpoint['loss']
    
    global_step = 0
    for epoch in range(resumed_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        tr_loss, global_step = train_epoch(epoch, args, model, teacher_models, train_dataloader, device, n_gpu, optimizer,
                                            scheduler, global_step, local_rank=args.local_rank)
        if args.rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

            output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

            ## Run on val dataset, this process is *TIME-consuming*.
            # logger.info("Eval on val dataset")
            # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

            R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
            if best_score <= R1:
                best_score = R1
                best_output_model_file = output_model_file
            logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

if __name__ == "__main__":
    main()
