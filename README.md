# Holistic Features are almost Sufficient for Text-to-Video Retrieval

The official source code of our CVPR24 paper TeachCLIP, "[Holistic Features are almost Sufficient for Text-to-Video Retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_Holistic_Features_are_almost_Sufficient_for_Text-to-Video_Retrieval_CVPR_2024_paper.pdf)".

![](./images/teachclip.png)

## Environment

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

```shell
conda create -n TeachCLIP python==3.8 -y
conda activate TeachCLIP
git clone https://github.com/ruc-aimc-lab/TeachCLIP.git
cd TeachCLIP
pip install -r requirements.txt
```


## Data

### Data download

+ We provide annotations of five datasets and checkpoints of three teachers ([X-CLIP](https://github.com/xuguohai/X-CLIP), [TS2-Net](https://github.com/yuqi657/ts2_net) and [XPool](https://github.com/layer6ai-labs/xpool)) trained on five datasets at [Google drive](https://drive.google.com/drive/folders/1cU0ehXfucf4M5IyDRSxywBadCt1LyZWz?usp=sharing). Video captions and data splits are provided in `Annotations` and `VideoSet`.

+ For raw videos, you can refer to the guides from [CLIP4Clip: Data Preparing](https://github.com/ArrowLuo/CLIP4Clip?tab=readme-ov-file#data-preparing). Put the videos into the corresponding `VideoData` folder for each dataset. (It is recommended to use symbolic links.)

### Data organization

Before starting to run the code, please organize the downloaded data in the following format: (The `Models` and `FeatureData` folders will be automatically generated during training and testing, respectively.)

```shell
data
├── datasets
│   ├── msrvtt
│   │   ├── Annotations
│   │   │   ├── MSRVTT_data.json
│   │   │   ├── MSRVTT_JSFUSION_test.csv
│   │   │   └── ...
│   │   ├── FeatureData
│   │   ├── Models
│   │   │   └── msrvtt-7k_xclip+ts2net-as-teacher_vit32
│   │   │       ├── run0
│   │   │       └── ...
│   │   ├── QuerySet
│   │   │   ├── msrvtt1k-test-query.txt
│   │   │   ├── msrvtt3k-test-query.txt
│   │   │   └── ...
│   │   └── VideoData
│   │   │   ├── video0.mp4
│   │   │   ├── video1.mp4
│   │   │   └── ...
│   │   └── VideoSet
│   │       ├── msrvtt1k-test.txt
│   │       ├── msrvtt1k-train.txt
│   │       └── ...
│   ├── activitynet
│   ├── didemo
│   ├── msvd
│   └── vatex
└── teacher_checkpoints
    ├── xclip
    │   ├── didemo_xclip_model.bin
    │   ├── msrvtt-7k_xclip_model.bin
    │   └── ...
    ├── ts2net
    └── xpool
```

## Code

### Training

Write the config file before training. [Here](https://github.com/ruc-aimc-lab/TeachCLIP/tree/main/configs), we provide a demo config file for each dataset. You can train TeachCLIP on specified GPUs and dataset by using the following command (taking `msrvtt-9k` as an example):

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --config_path configs/msrvtt-9k.yaml
```

### Inference

Use the following command to extract video / text features:

```shell
bash do_extract_video_feat.sh $test_collection $videoset $model_name
# e.g. bash do_extract_video_feat.sh msrvtt msrvtt1k-test msrvtt/Models/msrvtt-9k_xclip+ts2net-as-teacher_vit32/run0

bash do_extract_text_feat.sh $test_collection $queryset $model_name
# e.g. bash do_extract_text_feat.sh msrvtt msrvtt1k-test-query msrvtt/Models/msrvtt-9k_xclip+ts2net-as-teacher_vit32/run0
```

### Evaluation

After obtaining the text and video features, the evaluation metrics can be calculated using the following instructions:

```shell
bash do_eval.sh $test_collection $text_feat_name $video_feat_name $gt_file_name
# e.g. bash do_eval.sh msrvtt msrvtt1k-test-query/msrvtt/msrvtt-9k_xclip+ts2net-as-teacher_vit32/run0 msrvtt1k-test/msrvtt/msrvtt-9k_xclip+ts2net-as-teacher_vit32/run0 msrvtt1k-gt
```

## Citation

If you find our method useful in your work, please cite:

```python
@inproceedings{teachclip,
  title = {Holistic Features are almost Sufficient for Text-to-Video Retrieval}
  author = {Tian, Kaibin and Zhao, Ruixiang and Xin, Zijie and Lan, Bangxiang and Li, Xirong},
  year = {2024},
  booktitle={CVPR}
}
```


## Acknowledgments

The implementation of TeachCLIP relies on resources from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip"), [X-CLIP](https://github.com/xuguohai/X-CLIP "X-CLIP") and [XPool](https://github.com/layer6ai-labs/xpool "XPool"). We thank the original authors for their open-sourcing.


## Contact

If you encounter any issue when running the code, please feel free to reach us either by creating a new issue in the GitHub or by emailing

- Ruixiang Zhao ([ruixiangzhao@ruc.edu.cn](mailto:ruixiangzhao@ruc.edu.cn))
