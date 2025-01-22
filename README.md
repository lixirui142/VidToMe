## VidToMe: Video Token Merging for Zero-Shot Video Editing (CVPR 2024)<br><sub>Official Pytorch Implementation</sub>

[Xirui Li](https://lixirui142.github.io/), [Chao Ma](https://vision.sjtu.edu.cn/), [Xiaokang Yang](https://english.seiee.sjtu.edu.cn/english/detail/842_802.htm), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<br>

[**Project Page**](https://vidtome-diffusion.github.io/) | [**Paper**](https://arxiv.org/abs/2312.10656) | [**Summary Video**](https://youtu.be/cZPtwcRepNY) | [**Model Card 🤗**](https://huggingface.co/jadechoghari/VidToMe)

Also check [VISION-SJTU/VidToMe](https://github.com/VISION-SJTU/VidToMe/)

https://github.com/lixirui142/VidToMe/assets/46120552/e1492b83-eb3c-440b-a47d-330995284e14

VidToMe merges similar self-attention tokens across frames, improving temporal consistency while reducing memory consumption.

<details><summary> Abstract </summary>

> *Diffusion models have made significant advances in generating high-quality images, but their application to video generation has remained challenging due to the complexity of temporal motion. Zero-shot video editing offers a solution by utilizing pre-trained image diffusion models to translate source videos into new ones. Nevertheless, existing methods struggle to maintain strict temporal consistency and efficient memory consumption. In this work, we propose a novel approach to enhance temporal consistency in generated videos by merging self-attention tokens across frames. By aligning and compressing temporally redundant tokens across frames, our method improves temporal coherence and reduces memory consumption in self-attention computations. The merging strategy matches and aligns tokens according to the temporal correspondence between frames, facilitating natural temporal consistency in generated video frames. To manage the complexity of video processing, we divide videos into chunks and develop intra-chunk local token merging and inter-chunk global token merging, ensuring both short-term video continuity and long-term content consistency. Our video editing approach seamlessly extends the advancements in image editing to video editing, rendering favorable results in temporal consistency over state-of-the-art methods.*
</details>

## Updates
- [10/2024] Diffusers Implementation 🧨 by [@jadechoghari](https://github.com/jadechoghari) HF 🤗.
- [02/2024] Code is released.
- [02/2024] Accepted to CVPR 2024!
- [12/2023] Release paper and website.

### TODO
- [ ] Release evaluation dataset and more examples.
- [ ] Release evaluation code.

## Diffusers Implementation - easy set up
[![VidToMe](https://img.shields.io/badge/%F0%9F%A4%97%20VidToMe-blue)](https://huggingface.co/jadechoghari/VidToMe)

## Setup

1. Clone the repository. 

```shell
git clone git@github.com:lixirui142/VidToMe.git
cd VidToMe
```

2. Create a new conda environment and install PyTorch following [PyTorch Official Site](https://pytorch.org/get-started/locally/). Then pip install required packages.

```shell
conda create -n vidtome python=3.9
conda activate vidtome
# Install torch, torchvision (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt
```

We recommand installing [xformers](https://github.com/facebookresearch/xformers) for fast and memory-efficient attention.

## Run

```shell
python run_vidtome.py --config configs/tea-pour.yaml
```

Check more config examples in ['configs'](configs). The default config value are specified in ['default.yaml'](configs/default.yaml) with explanation.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex

@inproceedings{li2024vidtome,
    title={VidToMe: Video Token Merging for Zero-Shot Video Editing},
    author={Li, Xirui and Ma, Chao and Yang, Xiaokang and Yang, Ming-Hsuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
    }

```

## Acknowledgments

The code is mainly developed based on [ToMeSD](https://github.com/dbolya/tomesd), [PnP](https://github.com/MichalGeyer/plug-and-play), [Diffusers](https://github.com/huggingface/diffusers).
