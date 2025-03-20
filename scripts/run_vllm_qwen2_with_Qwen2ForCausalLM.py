import tempfile
import torch
from typing import List

# from vllm import SamplingParams
from vllm.config import (VllmConfig,
                         ModelConfig,
                         SchedulerConfig,
                         CacheConfig,
                         ParallelConfig,
                         LoadConfig,
                         DeviceConfig,
                         get_current_vllm_config,
                         set_current_vllm_config)
from vllm.forward_context import set_forward_context, get_forward_context
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.distributed import (ensure_kv_transfer_initialized,
                              ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM, Qwen2TokenizerFast

from vllm_qwen2 import Qwen2ForCausalLM as CustomizedQwen2ForCausalLM


# def create_rope_embeddings(dim, seq_len, rope_theta, device="cuda"):
#     """
#     Creates RoPE embeddings.
#
#     Args:
#         dim: Embedding dimension.
#         seq_len: Maximum sequence length.
#         device: Device to create embeddings on.
#
#     Returns:
#         A tensor containing precomputed rotation values.
#     """
#     freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
#     freqs = freqs.to(device)
#     t = torch.arange(seq_len, device=device)
#     freqs_t = torch.outer(t, freqs).float()
#     sin = torch.sin(freqs_t)
#     cos = torch.cos(freqs_t)
#     return torch.stack([cos, -sin, sin, cos], dim=-1).view(seq_len, dim // 2, 2, 2)


def run_gpt():
    device = "cuda"  # the device to load the model onto

    prompts = ["Give me a short introduction to large language model.",
               "Give me a short introduction to large language model."]
    # Tokenizer
    tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2-7B-Instruct")
    # prompt = prompts[0]
    # messages = [{"role": "user", "content": prompt}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(prompts, return_tensors="pt").to(device)
    print(f"====== model_inputs shape: {model_inputs.input_ids.shape} =====")
    print(f"===== model_inputs: {model_inputs} =====")
    b, s = model_inputs.input_ids.shape

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    seq_lens: List[int] = []
    for i in range(1):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData.from_seqs(model_inputs["input_ids"][0])},
                sampling_params=sampling_params,
                block_tables={0: [1]},
            ))
        seq_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seq_lens,
        query_lens=seq_lens,
        device=device,
        pin_memory=True)

    # Create a model. Qwen/Qwen2-7B-Instruct or Qwen/Qwen2.5-0.5B-Instruc
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_model = HFQwen2ForCausalLM.from_pretrained(model_name)
    pretrained_config = hf_model.config

    # Reference https://docs.vllm.ai/en/latest/design/huggingface_integration.html#huggingface-integration
    # tests/lora/test_worker.py
    model_config = ModelConfig(
        model=model_name,
        task="generate",
        tokenizer=model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    load_config = LoadConfig(
        download_dir=None,
        load_format="dummy",
    )
    parallel_config = ParallelConfig(1, 1, False)
    scheduler_config = SchedulerConfig("generate", 32, 32, 32)
    device_config = DeviceConfig("cuda")
    cache_config = CacheConfig(block_size=16,
                               gpu_memory_utilization=1.,
                               swap_space=0,
                               cache_dtype="auto")
    # lora_config = LoRAConfig(max_lora_rank=8, max_cpu_loras=32,
    #                          max_loras=32),
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        load_config=load_config,
        device_config=device_config, )
    vllm_config = vllm_config.with_hf_config(pretrained_config)

    # vllm/worker/worker.py
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
    init_distributed_environment(parallel_config.world_size, 0,
                                 f"file://{tempfile.mkstemp()[1]}", 0)
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
    ensure_kv_transfer_initialized(vllm_config)

    with set_current_vllm_config(vllm_config):
        vllm_model = CustomizedQwen2ForCausalLM(vllm_config=vllm_config)  # prefix is "" for top level model
        print(f"===== vllm_config: {vllm_config} =====")
        # <class 'vllm.model_executor.models.qwen2.Qwen2ForCausalLM'>
        print(f"===== Model CLS: {type(vllm_model)} =====")

        vllm_model.load_weights(hf_model.state_dict().items())
        vllm_model.to(device).eval()
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.

        # https://github.com/vllm-project/vllm/blob/8310e0b59b412693831e7b8e6fa11fc13698e17c/vllm/worker/model_runner.py#L1655

        # positions must have shape [num_tokens] or [batch_size, seq_len]
        positions = torch.arange(s, dtype=torch.long, device=device).repeat(2, 1)
        print(f"====== positions shape: {positions.shape} =====")
        print(f"===== positions: {positions} =====")

        # [Remove unused kwargs from model definitions (#13555)](https://github.com/vllm-project/vllm/commit/cdc1fa12eb1ba4795d24e97dcffa2018668a9267)
        attention_metadata = AttentionMetadata(
            num_prefills=b,
            num_prefill_tokens=s,
            num_decode_tokens=0,
            slot_mapping=torch.zeros(1),
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )

        kv_caches = [
            torch.tensor([], dtype=torch.bfloat16, device=device)
            for _ in range(pretrained_config.num_hidden_layers)
        ]

        with set_forward_context(attention_metadata, vllm_config):
            forward_context = get_forward_context()
            print(f"===== forward_context: {forward_context} =====")
            current_vllm_config = get_current_vllm_config()
            print(f"===== current_vllm_config: {current_vllm_config} =====")
            hidden_states = vllm_model(model_inputs.input_ids, positions, kv_caches, attention_metadata)

        logits = vllm_model.compute_logits(hidden_states, sampling_metadata)

        # TODO: This is only 1 token, need to decode iteratively
        outputs = vllm_model.sample(logits, sampling_metadata)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # Prompt: 'Give me a short introduction to large language model.', Generated text: ' A large language model, often referred to as a language model, is a type'
        print(
            f"===== Prompt: {prompt!r}, Generated text: {generated_text!r} =====")

    return outputs


if __name__ == "__main__":
    run_gpt()
