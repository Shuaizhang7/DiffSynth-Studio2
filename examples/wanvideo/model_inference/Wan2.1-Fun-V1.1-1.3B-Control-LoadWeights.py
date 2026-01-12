"""
Wan1.3B 模型权重加载脚本

此脚本演示如何加载 PAI/Wan2.1-Fun-V1.1-1.3B-Control 模型的权重

模型地址: https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control/files
"""

import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def load_wan_1_3b_model(torch_dtype=torch.bfloat16, device="cuda"):
    """
    加载 Wan1.3B 模型权重

    Args:
        torch_dtype: 模型权重的数据类型，默认 bfloat16
        device: 运行设备，默认 "cuda"

    Returns:
        WanVideoPipeline: 加载好的模型管道
    """
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            # 扩散模型权重
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="diffusion_pytorch_model*.safetensors"
            ),
            # T5 文本编码器
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
            ),
            # VAE 模型
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="Wan2.1_VAE.pth"
            ),
            # CLIP 图像编码器
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            ),
        ],
        # 分词器配置
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/"
        ),
    )
    return pipe


def check_model_weights():
    """
    检查模型权重是否正确加载
    """
    print("正在加载 Wan1.3B 模型权重...")

    pipe = load_wan_1_3b_model()

    print("\n模型组件加载状态:")
    print(f"  - text_encoder: {type(pipe.text_encoder).__name__}")
    print(f"  - dit: {type(pipe.dit).__name__}")
    print(f"  - vae: {type(pipe.vae).__name__}")
    print(f"  - image_encoder: {type(pipe.image_encoder).__name__}")
    print(f"  - motion_controller: {type(pipe.motion_controller).__name__}")

    # 获取模型参数量
    total_params = sum(p.numel() for p in pipe.dit.parameters()) / 1e9
    print(f"\nDiT 模型参数量: {total_params:.2f}B")

    # 获取 VAE 参数量
    vae_params = sum(p.numel() for p in pipe.vae.parameters()) / 1e6
    print(f"VAE 模型参数量: {vae_params:.2f}M")

    print("\n模型权重加载成功!")
    return pipe


if __name__ == "__main__":
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU 运行 (可能很慢)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    pipe = check_model_weights()
