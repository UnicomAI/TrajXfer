
# TrajXfer

## 🧭 Description
TrajXfer is a **trajectory-based acceleration method** for diffusion and flow-based generative models.
It can be viewed as an **advanced version of Shortest Path Diffusion ([ShortDF](https://github.com/UnicomAI/ShortDF))**, focusing on transferring optimized sampling trajectories to enable efficient few-step inference.

TrajXfer aims to **transfer the visual fidelity of long sampling trajectories into extreme few-step generation** via a lightweight, trajectory-aware LoRA, without modifying the semantic capacity of the base model. The full methodology, theoretical analysis, and extensive experiments are described in a research work that is **currently under submission**.

Below is a qualitative illustration of **trajectory transfer**, showing how the visual quality of long-trajectory sampling can be preserved under few-step inference using TrajXfer.

* Baseline: FLUX.1-dev with standard multi-step sampling
* **TrajXfer(Ours)**: FLUX.1-dev with TrajXfer-enabled few-step sampling
<img width="2055" height="535" alt="image" src="https://github.com/user-attachments/assets/003a607b-d626-4ac1-92b1-cff3930c14fe" />


---

## 🚀 Quick Start

TrajXfer can be directly used with the **x-flux pipeline**.
The training and implementation are based on x-flux, so using TrajXfer together with x-flux is recommended.
Other frameworks (e.g., diffusers-based pipelines) can also be adapted if needed.

Example Python usage:

```
from src.flux.xflux_pipeline import XFluxPipeline

device = 'cuda:0'
xflux_pipeline = XFluxPipeline('flux-dev', device, False)

prompts=["ultra-photorealistic, natural sweet young woman, early 20s, candid gentle smile, soft and clear skin with visible pores and natural light freckles, subtle natural makeup, silky slightly wavy medium-length hair, fresh and lively expression, captured by a Leica SL2-S with a Summilux-M 50mm f/1.4 lens, natural outdoor sunlight (golden hour), delicate shallow depth of field (bokeh background), cinematic warm tones, slight wind blowing the hair, extremely detailed facial texture, true-to-life color grading, authentic human proportions, no anime, no illustration, no cgi, no 3d, no painting, no digital art, pure real-world photography style."]

base_result = xflux_pipeline(
                prompt=prompts,
                width=1024,
                height=1024,
                guidance=4,
                num_steps=30,
                seed=123456789,
                true_gs=3.5,
                neg_prompt="",
                timestep_to_start_cfg=5,
            )



lora_name = 'trajxfer'
local_path = 'trajxfer_flux1.0-dev_lora.safetensors'

#10 steps
xflux_pipeline.set_lora(local_path=local_path, name=lora_name, lora_weight=0.8)

trajxfer_result = xflux_pipeline(
    prompt=prompts,
    width=1024,
    height=1024,
    guidance=4,
    num_steps=10,
    seed=123456789,
    true_gs=3.5,
    neg_prompt="",
    timestep_to_start_cfg=5,
)

#2 steps
xflux_pipeline.set_lora(local_path=local_path, name=lora_name, lora_weight=1)

trajxfer_result_2 = xflux_pipeline(
    prompt=prompts,
    width=1024,
    height=1024,
    guidance=4,
    num_steps=2,
    seed=123456789,
    true_gs=3.5,
    neg_prompt="",
    timestep_to_start_cfg=5,
)
```

This example realizes few-step inference with visual characteristics matching long sampling trajectories.

**Notes:**

* When increasing the number of sampling steps, reducing `lora_weight` may yield better performance
* Style control and output diversity can be maintained through prompt design

---

## 📦 Weights

TrajXfer is designed for **large-scale generative foundation models**, whose parameter counts typically reach **hundreds of billions**.

To use TrajXfer:

1. Download the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) base model

2. Download the [TrajXfer LoRA weights](https://drive.google.com/file/d/1nmtqgK_abJo-KFpAlY57nsT2iglDadO4/view?usp=drive_link)
---

## 🌟 Acknowledgements

This implementation is built on top of the [**x-flux**](https://github.com/XLabs-AI/x-flux) project.
We sincerely thank the **XLabs-AI** team for their open-source contributions to the FLUX ecosystem.

---



## 📖 Citation

If you find this work useful, please consider citing the following works.


```
@inproceedings{chen2025optimizing,
  title     = {Optimizing for the Shortest Path in Denoising Diffusion Models},
  author    = {Chen, Ping and Zhang, Xingpeng and Liu, Zhaoxiang and Hu, Huan and Liu, Xiang and Wang, Kai and Wang, Min and Qian, Yanlin and Lian, Shiguo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {18021--18030},
  year      = {2025}
}


@article{trajxfer2025,
  title   = {TrajXfer},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2025}
}
```

---
