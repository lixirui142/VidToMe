# VidToMe: Video Token Merging for Zero-Shot Video Editing

<!--https://github.com/williamyang1991/Rerender_A_Video/assets/18130694/82c35efb-e86b-4376-bfbe-6b69159b8879-->

[Xirui Li](https://github.com/lixirui142), [Chao Ma](https://vision.sjtu.edu.cn/), [Xiaokang Yang](https://english.seiee.sjtu.edu.cn/english/detail/842_802.htm) and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<br>

[**Project Page**](https://vidtome-diffusion.github.io/) | [**Paper**] | [**Video Summary**](https://youtu.be/MgkVFEt3fCk)

> **Abstract:** *Large text-to-image diffusion models have exhibited impressive proficiency in generating high-quality images. However, when applying these models to video domain, ensuring temporal consistency across video frames remains a formidable challenge. This paper proposes a novel zero-shot text-guided video-to-video translation framework to adapt image models to videos. The framework includes two parts: key frame translation and full video translation. The first part uses an adapted diffusion model to generate key frames, with hierarchical cross-frame constraints applied to enforce coherence in shapes, textures and colors. The second part propagates the key frames to other frames with temporal-aware patch matching and frame blending. Our framework achieves global style and local texture temporal consistency at a low cost (without re-training or optimization). The adaptation is compatible with existing image diffusion techniques, allowing our framework to take advantage of them, such as customizing a specific subject with LoRA, and introducing extra spatial guidance with ControlNet. Extensive experimental results demonstrate the effectiveness of our proposed framework over existing methods in rendering high-quality and temporally-coherent videos.*
