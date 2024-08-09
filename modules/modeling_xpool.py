"""
Adapted from: https://github.com/layer6ai-labs/xpool
"""
import torch
from torch import nn

from modules.transformer_xpool import Transformer

from modules.until_module import AllGather
allgather = AllGather.apply

class XPool(nn.Module):
    def __init__(self, task_config=None):
        super(XPool, self).__init__()
        self.task_config = task_config # 只使用里面的world_size, rank
        self.huggingface = True
        self.max_frames = 12
        
        if self.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.pool_frames = Transformer()


    def get_origin_text_and_video_features(self, input_ids, attention_mask, video):
        batch_size = video.shape[0]
        text_data = {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze()}
        video_data = video
        video_data = video_data.reshape(-1, 3, 224, 224)
        
        if self.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data.float())
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)

        video_features = video_features.reshape(batch_size, self.max_frames, -1)

        return text_features, video_features


    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,return_fine=False):
        text_features, video_features = self.get_origin_text_and_video_features(input_ids, attention_mask, video)
        
        if self.training:
            text_features = allgather(text_features, self.task_config)
            video_features = allgather(video_features, self.task_config)
            torch.distributed.barrier()
        else:
            text_features = allgather(text_features, self.task_config)
            video_features = allgather(video_features, self.task_config)
            torch.distributed.barrier()

        video_features_pooled, frame_attention_weights = self.pool_frames(text_features, video_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        video_features_pooled = video_features_pooled / video_features_pooled.norm(dim=-1, keepdim=True)

        video_features_pooled = video_features_pooled.permute(1,2,0)
        text_features = text_features.unsqueeze(1)
        
        logit_scale = self.clip.logit_scale.exp()
        sims = logit_scale * torch.bmm(text_features, video_features_pooled).squeeze(1)

        if return_fine:
            return sims,frame_attention_weights
        else:
            return sims


