## TransPixeler: Advancing Text-to-Video Generation with Transparency (CVPR2025)
<br>
    <a href="https://arxiv.org/abs/2501.03006"><img src='https://img.shields.io/badge/arXiv-2501.03006-b31b1b.svg'></a>
    <a href='https://wileewang.github.io/TransPixar'><img src='https://img.shields.io/badge/Project_Page-TransPixar-blue'></a>
    <a href='https://huggingface.co/spaces/wileewang/TransPixar'><img src='https://img.shields.io/badge/HuggingFace-TransPixar-yellow'></a>
    <a href="https://discord.gg/7Xds3Qjr"><img src="https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp"></a>
    <a href="https://github.com/wileewang/TransPixar/blob/main/wechat_group.jpg"><img src="https://img.shields.io/badge/Wechat-Join-green?logo=wechat&amp"></a>
<!--     <a href='https://www.youtube.com/watch?v=Wq93zi8bE3U'><img src='https://img.shields.io/badge/Demo_Video-MotionDirector-red'></a> -->
<br>

[Luozhou Wang*](https://wileewang.github.io/), 
[Yijun Li**](https://yijunmaverick.github.io/), 
[Zhifei Chen](), 
[Jui-Hsien Wang](http://juiwang.com/), 
[Zhifei Zhang](https://zzutk.github.io/), 
[He Zhang](https://sites.google.com/site/hezhangsprinter), 
[Zhe Lin](https://sites.google.com/site/zhelin625/home), 
[Ying-Cong Chen†](https://www.yingcong.me)

HKUST(GZ), HKUST, Adobe Research.

\* Internship Project
\** Project Lead
† Corresponding Author

Text-to-video generative models have made significant strides, enabling diverse applications in entertainment, advertising, and education. However, generating RGBA video, which includes alpha channels for transparency, remains a challenge due to limited datasets and the difficulty of adapting existing models. Alpha channels are crucial for visual effects (VFX), allowing transparent elements like smoke and reflections to blend seamlessly into scenes.
We introduce TransPixar, a method to extend pretrained video models for RGBA generation while retaining the original RGB capabilities. TransPixar leverages a diffusion transformer (DiT) architecture, incorporating alpha-specific tokens and using LoRA-based fine-tuning to jointly generate RGB and alpha channels with high consistency. By optimizing attention mechanisms, TransPixeler preserves the strengths of the original RGB model and achieves strong alignment between RGB and alpha channels despite limited training data.
Our approach effectively generates diverse and consistent RGBA videos, advancing the possibilities for VFX and interactive content creation.

<!-- insert a teaser gif -->
<!-- <img src="assets/mi.gif"  width="640" /> -->



## 📰 News
* **[2025.02.26]** **TransPixeler** is accepted by CVPR 2025! See you in Nashville!
* **[2025.01.19]** We've renamed our project from **TransPixar** to **TransPixeler**!!
* **[2025.01.17]** We’ve created a [Discord group](https://discord.gg/7Xds3Qjr) and a [WeChat group](https://github.com/wileewang/TransPixar/blob/main/wechat_group.jpg)! Everyone is welcome to join for discussions and collaborations. Let’s work together to make the repository even better!
* **[2025.01.14]** Our repository has been receiving significant attention recently, and we’re thrilled by the interest in TransPixar! Many users have requested deployments on new video models, including Hunyuan and LTX, as well as support for ComfyUI. We’ve added these to our to-do list and are eager to make progress. However, training TransPixar LoRA for different video models requires substantial resources and time, so we kindly ask for your patience. Stay tuned for updates! Additionally, we warmly welcome contributions to this repository—your support makes a difference!
* **[2025.01.07]** We have released the project page, arXiv paper, inference code and huggingface demo for TransPixar + CogVideoX-5B.



## 🚧 Todo List
* [x] Release code, paper and demo.
* [x] Release checkpoints of joint generation (RGB + Alpha).
* [ ] Release checkpoints for Mochi and CogVideoX-I2V
* [ ] Provide support for ComfyUI
* [ ] Deploy TransPixar on Hunyuan and LTX video models
<!-- * [ ] Release checkpoints of more modalities (RGB + Depth).
* [ ] Release checkpoints of conditional generation (RGB->Alpha). -->


## Contents

* [Installation](#installation)
* [TransPixar LoRA Hub](#lora-hub) 
* [Training](#training)
* [Inference](#inference)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

<!-- * [Motion Embeddings Hub](#motion-embeddings-hub) -->

## Installation

```bash
conda create -n TransPixar python=3.10
conda activate TransPixar
pip install -r requirements.txt
```



## TransPixeler LoRA Hub

Our pipeline is designed to support various video tasks, including Text-to-RGBA Video, Image-to-RGBA Video.

We provide the following pre-trained LoRA weights for different tasks:

| Task          | Base Model                                                    | Frames | LoRA weights                                                       | Inference VRAM |
|---------------|---------------------------------------------------------------|--------|--------------------------------------------------------------------|----------------|
| T2V + RGBA   | [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview) | 37     | Coming soon                                                       | TBD            |
| T2V + RGBA   | [THUDM/CogVideoX-5B](https://huggingface.co/THUDM/CogVideoX-5b)       | 49     | [link](https://huggingface.co/wileewang/TransPixar/blob/main/cogvideox_rgba_lora.safetensors) | ~24GB          |
| I2V + RGBA   | [THUDM/CogVideoX-5b-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V) | 49     | Coming soon                                                       | TBD            |


## Training - RGB + Alpha Joint Generation
We have open-sourced the training code for **Mochi** on RGBA joint generation. Please refer to the [Mochi README](Mochi/README.md) for details.


## Inference - Gradio Demo
In addition to the [Hugging Face online demo](https://huggingface.co/spaces/wileewang/TransPixar), users can also launch a local inference demo based on CogVideoX-5B by running the following command:

```bash
python app.py
```

## Inference - Command Line Interface (CLI)
To generate RGBA videos, navigate to the corresponding directory for the video model and execute the following command:
```bash
python cli.py \
    --lora_path /path/to/lora \
    --prompt "..." \

```



## Acknowledgement

* [finetrainers](https://github.com/a-r-r-o-w/finetrainers): We followed their implementation of Mochi training and inference.
* [CogVideoX](https://github.com/THUDM/CogVideo): We followed their implementation of CogVideoX training and inference.

We are grateful for their exceptional work and generous contribution to the open-source community.

## Citation

 ```bibtex
@misc{wang2025transpixeler,
      title={TransPixeler: Advancing Text-to-Video Generation with Transparency}, 
      author={Luozhou Wang and Yijun Li and Zhifei Chen and Jui-Hsien Wang and Zhifei Zhang and He Zhang and Zhe Lin and Yingcong Chen},
      year={2025},
      eprint={2501.03006},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.03006}, 
}
``` 

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wileewang/TransPixar&type=Date)](https://star-history.com/#wileewang/TransPixar&Date)
