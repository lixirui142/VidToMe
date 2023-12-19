# VidToMe: Video Token Merging for Zero-Shot Video Editing

<!--https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/82c35efb-e86b-4376-bfbe-6b69159b8879-->

[Xirui Li](https://github.com/lixirui142), [Chao Ma](https://vision.sjtu.edu.cn/), [Xiaokang Yang](https://english.seiee.sjtu.edu.cn/english/detail/842_802.htm), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<br>

[**Project Page**](https://vidtome-diffusion.github.io/) | [**Paper**](https://arxiv.org/abs/2312.10656) | [**Summary Video**](https://youtu.be/cZPtwcRepNY)

https://github.com/lixirui142/VidToMe/assets/46120552/d44b086f-e566-41c6-a83a-c154a68a3307

> **Abstract:** *Diffusion models have made significant advances in generating high-quality images, but their application to video generation has remained challenging due to the complexity of temporal motion. Zero-shot video editing offers a solution by utilizing pre-trained image diffusion models to translate source videos into new ones. Nevertheless, existing methods struggle to maintain strict temporal consistency and efficient memory consumption. In this work, we propose a novel approach to enhance temporal consistency in generated videos by merging self-attention tokens across frames. By aligning and compressing temporally redundant tokens across frames, our method improves temporal coherence and reduces memory consumption in self-attention computations. The merging strategy matches and aligns tokens according to the temporal correspondence between frames, facilitating natural temporal consistency in generated video frames. To manage the complexity of video processing, we divide videos into chunks and develop intra-chunk local token merging and inter-chunk global token merging, ensuring both short-term video continuity and long-term content consistency. Our video editing approach seamlessly extends the advancements in image editing to video editing, rendering favorable results in temporal consistency over state-of-the-art methods.*
