import argparse

from .main import run_server


def main():
    parser = argparse.ArgumentParser(description="InfiniteTalk Serving (LightX2V-compatible API)")

    # Model paths (mirrors generate_infinitetalk.py)
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to Wan2.1-I2V checkpoint directory")
    parser.add_argument("--wav2vec_dir", type=str, required=True, help="Path to wav2vec checkpoint directory")
    parser.add_argument("--infinitetalk_dir", type=str, required=True, help="Path to infinitetalk.safetensors")
    parser.add_argument("--quant_dir", type=str, default=None, help="Path to quant checkpoint file (optional)")
    parser.add_argument("--quant", type=str, default=None, help="Quantization type: int8 or fp8 (optional)")
    parser.add_argument("--dit_path", type=str, default=None, help="Optional merged dit checkpoint path")

    # LoRA (optional)
    parser.add_argument(
        "--lora_dir",
        type=str,
        nargs="+",
        default=None,
        help="LoRA checkpoint file path(s) (.safetensors). Accepts multiple values.",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="LoRA scale(s). Must match --lora_dir count if multiple are provided.",
    )

    # Runtime
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--max_queue_size", type=int, default=10, help="Max pending+running tasks")
    parser.add_argument(
        "--print_timing",
        action="store_true",
        default=False,
        help="Print wall-clock timing for major stages (uses print + flush). "
        "Also enabled by env INFINI_PRINT_TIMING=1",
    )

    # Defaults for generation (can be overridden per request with optional fields)
    parser.add_argument("--size", type=str, default="infinitetalk-480", help="infinitetalk-480 or infinitetalk-720")
    parser.add_argument("--motion_frame", type=int, default=9, help="Motion frame length for streaming continuation")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift (default depends on size)")
    parser.add_argument("--sample_text_guide_scale", type=float, default=5.0, help="Text CFG scale")
    parser.add_argument("--sample_audio_guide_scale", type=float, default=4.0, help="Audio CFG scale")
    parser.add_argument("--offload_model", action="store_true", help="Enable CPU offload between forwards")
    parser.add_argument("--max_frame_num", type=int, default=1000, help="Default max frames in streaming mode")

    # Dist / VRAM
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None)

    args, _unknown = parser.parse_known_args()
    run_server(args)


if __name__ == "__main__":
    main()

