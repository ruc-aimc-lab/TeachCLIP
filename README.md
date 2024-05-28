# Holistic Features are almost Sufficient for Text-to-Video Retrieval

The official source code of our CVPR24 paper TeachCLIP, "[Holistic Features are almost Sufficient for Text-to-Video Retrieval](https://www.researchgate.net/publication/379270657_Holistic_Features_are_almost_Sufficient_for_Text-to-Video_Retrieval)".

![](./images/teachclip.png)

## Requirement

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

```shell
conda create -n TeachCLIP python==3.8 -y
conda activate TeachCLIP
git clone https://github.com/ruc-aimc-lab/TeachCLIP.git
cd TeachCLIP
pip install -r requirements.txt
```


## How to Run

### Data download

+ We provide data splits of five datasets at [Google drire](https://drive.google.com/drive/folders/1wfx0N0IyHkEwHWy5PYCij2i7kXynipSL?usp=sharing).

+ For raw videos, you can refer to the guides from [CLIP4Clip: Data Preparing](https://github.com/ArrowLuo/CLIP4Clip?tab=readme-ov-file#data-preparing).

### Teacher checkpoint download

We offer checkpoints of three teachers ([X-CLIP](https://github.com/xuguohai/X-CLIP), [TS2-Net](https://github.com/yuqi657/ts2_net) and [XPool](https://github.com/layer6ai-labs/xpool)) trained on five datasets. Please download the corresponding checkpoints before you start training the student model through knowledge distillation.

+ [Google drive](https://drive.google.com/drive/folders/1qaA8ObtQa8wbpfCyHcrh8MOk_W05VRR3?usp=sharing)

### Training and evaluation

| Dataset   | Training command                            | Evaluation command                         |
| --------- | ------------------------------------------- | ------------------------------------------ |
| MSRVTT-1k | `bash scripts/train_teachclip_msrvtt-1k.sh` | `bash scripts/eval_teachclip_msrvtt-1k.sh` |
| MSRVTT-3k | `bash scripts/train_teachclip_msrvtt-3k.sh` | `bash scripts/eval_teachclip_msrvtt-3k.sh` |
| MSVD      | `bash scripts/train_teachclip_msvd.sh`      | `bash scripts/eval_teachclip_msvd.sh`      |
| VATEX     | `bash scripts/train_teachclip_vatex.sh`     | `bash scripts/eval_teachclip_vatex.sh`     |
| ActNetCap | `bash scripts/train_teachclip_actnet.sh`    | `bash scripts/eval_teachclip_actnet.sh`    |
| DiDeMo    | `bash scripts/train_teachclip_didemo.sh`    | `bash scripts/eval_teachclip_didemo.sh`    |

+ Fill in missing parameters (enclosed in `[ ]`) with your own path.

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
