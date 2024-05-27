import pandas as pd
import json

keys = []
video_ids = []
sentences = []

original_data_path = '/data3/zrx/codes/TeachCLIP/datasets/msrvtt_data/val_list_full.txt'
val_video_ids = []
with open(original_data_path, 'r') as f:
    for line in f:
        video_id = line.strip()
        val_video_ids.append(video_id)

all_data = json.load(open('/data3/zrx/codes/TeachCLIP/datasets/msrvtt_data/MSRVTT_data.json'))

i = 0
for sentence in all_data['sentences']:
    if sentence['video_id'] in val_video_ids:
        keys.append('ret'+str(i))
        video_ids.append(sentence['video_id'])
        sentences.append(sentence['caption'])
        i += 1

data = {'key': keys, 'video_id': video_ids, 'sentence': sentences}
pd.DataFrame(data).to_csv('/data3/zrx/codes/TeachCLIP/datasets/msrvtt_data/MSRVTT_full_split_val.csv', index=False)