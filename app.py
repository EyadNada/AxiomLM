"""
AxiomLM Minimalist Interactive Web Interface.
Industrial-grade, clean, low-latency generation engine with real-time telemetry.
"""

import os
import sys
import time
from typing import Generator, Tuple, Optional
import tiktoken
import torch
import gradio as gr

# Ensure repo root is on sys.path and alias GPTConfig for robust unpickling
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import __main__
from brain.train_gpt2 import (
    GPT,
    GPTConfig,
    sample_logits,
)
if not hasattr(__main__, "GPTConfig"):
    setattr(__main__, "GPTConfig", GPTConfig)

from brain.generate import load_model

# -----------------------------------------------------------------------------
# Global State & Device Auto-Detection
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICE_NAME = "Apple Silicon (MPS)"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

ENCODER = tiktoken.get_encoding("gpt2")

# Model cache: key -> model
_MODEL_CACHE = {}


def get_or_load_model(source_type: str, checkpoint_path: str, arch: str) -> GPT:
    """Retrieves cached model or loads from disk/HuggingFace."""
    cache_key = f"{source_type}:{checkpoint_path}:{arch}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if source_type == "pretrained_gpt2":
        model, _ = load_model(checkpoint_path=None, pretrained="gpt2", arch="classic", device=DEVICE)
    else:
        actual_path = checkpoint_path.strip()
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"Checkpoint file not found: {actual_path}")
        model, _ = load_model(checkpoint_path=actual_path, pretrained=None, arch=arch, device=DEVICE)

    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


# -----------------------------------------------------------------------------
# Streaming Generation Engine
# -----------------------------------------------------------------------------
def stream_inference(
    prompt: str,
    source_type: str,
    custom_checkpoint: str,
    arch: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    use_kv_cache: bool,
) -> Generator[Tuple[str, str], None, None]:
    """Generates autoregressive tokens and streams output with live latency telemetry."""
    if not prompt or not prompt.strip():
        yield "", "Error: Prompt cannot be empty."
        return

    checkpoint_target = custom_checkpoint if custom_checkpoint.strip() else "checkpoints/model_latest.pt"

    try:
        model = get_or_load_model(source_type, checkpoint_target, arch)
    except Exception as err:
        yield "", f"Model Load Error: {str(err)}"
        return

    # Encode prompt
    input_ids = ENCODER.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated_tokens = x.clone()

    full_text = prompt
    tokens_generated = 0
    t_start = time.perf_counter()

    k_val = int(top_k) if top_k > 0 else None
    p_val = float(top_p) if top_p < 1.0 else None
    min_p_val = float(min_p) if min_p > 0.0 else None
    rep_val = float(repetition_penalty) if repetition_penalty > 1.0 else None

    # Prefill Phase
    with torch.no_grad():
        if use_kv_cache:
            kv_caches = [None] * model.config.n_layer
            logits, _, kv_caches = model(x, kv_caches=kv_caches)
            next_token = sample_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=k_val,
                top_p=p_val,
                min_p=min_p_val,
                repetition_penalty=rep_val,
                prev_tokens=x,
            )
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            token_str = ENCODER.decode([next_token.item()])
            full_text += token_str
            tokens_generated += 1

            t_now = time.perf_counter()
            dt = t_now - t_start
            throughput = tokens_generated / dt if dt > 0 else 0.0
            latency = (dt / tokens_generated) * 1000.0 if tokens_generated > 0 else 0.0
            telemetry = (
                f"Generated: {tokens_generated:3d} tokens | "
                f"Latency: {latency:5.1f} ms/tok | "
                f"Throughput: {throughput:5.1f} tok/s | "
                f"Engine: KV-Cache O(1) | "
                f"Device: {DEVICE_NAME}"
            )
            yield full_text, telemetry

            # Decode Phase
            while tokens_generated < max_tokens:
                logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
                next_token = sample_logits(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=k_val,
                    top_p=p_val,
                    min_p=min_p_val,
                    repetition_penalty=rep_val,
                    prev_tokens=generated_tokens,
                )
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                token_str = ENCODER.decode([next_token.item()])
                full_text += token_str
                tokens_generated += 1

                t_now = time.perf_counter()
                dt = t_now - t_start
                throughput = tokens_generated / dt if dt > 0 else 0.0
                latency = (dt / tokens_generated) * 1000.0 if tokens_generated > 0 else 0.0
                telemetry = (
                    f"Generated: {tokens_generated:3d} tokens | "
                    f"Latency: {latency:5.1f} ms/tok | "
                    f"Throughput: {throughput:5.1f} tok/s | "
                    f"Engine: KV-Cache O(1) | "
                    f"Device: {DEVICE_NAME}"
                )
                yield full_text, telemetry

                if next_token.item() == ENCODER.eot_token:
                    break
        else:
            # Naive Eager Recomputation
            while tokens_generated < max_tokens:
                logits, _ = model(generated_tokens)
                next_token = sample_logits(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=k_val,
                    top_p=p_val,
                    min_p=min_p_val,
                    repetition_penalty=rep_val,
                    prev_tokens=generated_tokens,
                )
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                token_str = ENCODER.decode([next_token.item()])
                full_text += token_str
                tokens_generated += 1

                t_now = time.perf_counter()
                dt = t_now - t_start
                throughput = tokens_generated / dt if dt > 0 else 0.0
                latency = (dt / tokens_generated) * 1000.0 if tokens_generated > 0 else 0.0
                telemetry = (
                    f"Generated: {tokens_generated:3d} tokens | "
                    f"Latency: {latency:5.1f} ms/tok | "
                    f"Throughput: {throughput:5.1f} tok/s | "
                    f"Engine: Naive Eager O(T^2) | "
                    f"Device: {DEVICE_NAME}"
                )
                yield full_text, telemetry

                if next_token.item() == ENCODER.eot_token:
                    break

    # Final summary telemetry
    t_end = time.perf_counter()
    dt_total = t_end - t_start
    throughput_final = tokens_generated / dt_total if dt_total > 0 else 0.0
    latency_final = (dt_total / tokens_generated) * 1000.0 if tokens_generated > 0 else 0.0
    engine_name = "KV-Cache O(1)" if use_kv_cache else "Naive Eager O(T^2)"
    telemetry_final = (
        f"Completed: {tokens_generated} tokens in {dt_total:.2f}s | "
        f"Average Latency: {latency_final:.1f} ms/token | "
        f"Throughput: {throughput_final:.1f} tokens/s | "
        f"Engine: {engine_name} | "
        f"Hardware: {DEVICE_NAME}"
    )
    yield full_text, telemetry_final


# -----------------------------------------------------------------------------
# Clean Minimalist CSS (macOS Light Aesthetic, Monospace Telemetry)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
/* Global Reset and Typography */
body, .gradio-container {
    background-color: #ffffff !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #111827 !important;
}

/* Header Styling */
.header-container {
    padding: 18px 0 24px 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
.header-title {
    font-size: 24px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #0f172a;
    margin: 0;
}
.header-desc {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}

/* Textboxes & Inputs */
textarea, input[type="text"] {
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    background-color: #ffffff !important;
    color: #111827 !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #0f172a !important;
    box-shadow: 0 0 0 1px #0f172a !important;
}

/* Primary Action Button */
button.primary-btn {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #0f172a !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: background-color 0.15s ease !important;
}
button.primary-btn:hover {
    background-color: #334155 !important;
    border-color: #334155 !important;
}

/* Secondary Button */
button.secondary-btn {
    background-color: #ffffff !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
}
button.secondary-btn:hover {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Telemetry Status Bar */
.telemetry-bar {
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace !important;
    font-size: 12px !important;
    color: #334155 !important;
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
}
"""


# -----------------------------------------------------------------------------
# Gradio Interface Definition
# -----------------------------------------------------------------------------
def build_app():
    theme = gr.themes.Default(
        primary_hue="slate",
        neutral_hue="slate",
    )
    with gr.Blocks(title="AxiomLM (124M)", css=CUSTOM_CSS, theme=theme) as demo:
        # Header
        gr.HTML(
            """
            <div class="header-container">
                <h1 class="header-title">AxiomLM</h1>
                <div class="header-desc">124M Parameter Autoregressive Pretraining & Inference Engine | PyTorch 2.x</div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Configuration & Parameters
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("### Model Configuration")
                    source_radio = gr.Radio(
                        choices=[
                            ("Local Checkpoint", "local"),
                            ("OpenAI Official Weights (gpt2)", "pretrained_gpt2"),
                        ],
                        value="local",
                        label="Model Source",
                    )
                    checkpoint_input = gr.Textbox(
                        value="checkpoints/model_latest.pt",
                        label="Checkpoint File Path",
                        placeholder="checkpoints/model_latest.pt",
                        visible=True,
                    )
                    arch_radio = gr.Radio(
                        choices=[
                            ("Modern (RoPE, RMSNorm, SwiGLU, GQA)", "modern"),
                            ("Classic (GPT-2 Baseline)", "classic"),
                        ],
                        value="modern",
                        label="Architecture Specification",
                    )

                with gr.Group():
                    gr.Markdown("### Sampling Parameters")
                    with gr.Row():
                        temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.8,
                            step=0.05,
                            label="Temperature",
                            info="0.0 = Greedy argmax",
                        )
                        max_tok_slider = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Max Tokens",
                        )

                    with gr.Row():
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p (Nucleus)",
                        )
                        min_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.05,
                            step=0.01,
                            label="Min-p Threshold",
                        )

                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Top-k",
                            info="0 = Disabled",
                        )
                        rep_slider = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.1,
                            step=0.05,
                            label="Repetition Penalty",
                        )

                    kv_cache_check = gr.Checkbox(
                        value=True,
                        label="Enable Key-Value (KV) Cache Acceleration (O(1) decoding)",
                    )

            # Right Column: Prompt, Generation & Telemetry
            with gr.Column(scale=6):
                prompt_box = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter prompt text here...",
                    lines=4,
                    value="Once upon a time in a quiet forest,",
                )

                with gr.Row():
                    generate_btn = gr.Button("Generate Text", elem_classes=["primary-btn"], scale=4)
                    stop_btn = gr.Button("Stop", elem_classes=["secondary-btn"], scale=1)
                    clear_btn = gr.Button("Clear", elem_classes=["secondary-btn"], scale=1)

                output_box = gr.Textbox(
                    label="Generated Output",
                    lines=10,
                    interactive=False,
                )

                telemetry_box = gr.Textbox(
                    label="Hardware & Latency Telemetry",
                    lines=1,
                    interactive=False,
                    elem_classes=["telemetry-bar"],
                    value=f"Ready | Execution Backend: {DEVICE_NAME}",
                )

                # Standard Prompt Presets
                gr.Examples(
                    examples=[
                        ["Once upon a time in a quiet forest,"],
                        ["The ancient astronomer looked through the telescope and discovered"],
                        ["Lily and her dog Max were walking down the street when"],
                        ["The fundamental theorem of arithmetic states that"],
                    ],
                    inputs=prompt_box,
                    label="Prompt Presets",
                )

        # Dynamic Visibility Handlers
        def on_source_change(source):
            if source == "local":
                return gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False, value="classic")

        source_radio.change(
            fn=on_source_change,
            inputs=[source_radio],
            outputs=[checkpoint_input, arch_radio],
        )

        # Execution Click Binding
        gen_event = generate_btn.click(
            fn=stream_inference,
            inputs=[
                prompt_box,
                source_radio,
                checkpoint_input,
                arch_radio,
                max_tok_slider,
                temp_slider,
                top_k_slider,
                top_p_slider,
                min_p_slider,
                rep_slider,
                kv_cache_check,
            ],
            outputs=[output_box, telemetry_box],
        )

        stop_btn.click(fn=None, cancels=[gen_event])
        clear_btn.click(fn=lambda: ("", "", f"Ready | Execution Backend: {DEVICE_NAME}"), outputs=[prompt_box, output_box, telemetry_box])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
