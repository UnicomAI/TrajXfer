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

trajxfer_result = xflux_pipeline(
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