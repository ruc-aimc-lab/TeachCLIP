# Holistic Features are almost Sufficient for Text-to-Video Retrieval

The official source code of our CVPR24 paper TeachCLIP, "[Holistic Features are almost Sufficient for Text-to-Video Retrieval](https://www.researchgate.net/publication/379270657_Holistic_Features_are_almost_Sufficient_for_Text-to-Video_Retrieval)".

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

+ We provide annotations of five datasets and checkpoints of three teachers ([X-CLIP](https://github.com/xuguohai/X-CLIP), [TS2-Net](https://github.com/yuqi657/ts2_net) and [XPool](https://github.com/layer6ai-labs/xpool)) trained on five datasets at [Google drive](https://drive.google.com/drive/folders/1cU0ehXfucf4M5IyDRSxywBadCt1LyZWz?usp=sharing). Data splits are provided in `annotations`.

+ For raw videos, you can refer to the guides from [CLIP4Clip: Data Preparing](https://github.com/ArrowLuo/CLIP4Clip?tab=readme-ov-file#data-preparing). Put the videos into the corresponding `video` folder for each dataset.

### Data organization

Before starting to run the code, please organize the downloaded data in the following format:

```shell
data
├── datasets
│   ├── msrvtt-9k
│   │   ├── annotations
│   │   │   ├── MSRVTT_data.json
│   │   │   ├── MSRVTT_JSFUSION_test.csv
│   │   │   └── MSRVTT_train.9k.csv
│   │   └── videos
│   │       ├── video0.mp4
│   │       ├── video1.mp4
│   │       └── ...
│   ├── activitynet
│   ├── didemo
│   ├── msrvtt-7k
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



### Inference



### Evaluation





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