from vllm import LLM, SamplingParams


def run_gpt():
    prompts = ["Give me a short introduction to large language model."]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM. Qwen/Qwen2-7B-Instruct or Qwen/Qwen2.5-0.5B-Instruct
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # <class 'vllm.model_executor.models.qwen2.Qwen2ForCausalLM'>
    print(
        f"============================= Model CLS: {llm.apply_model(lambda model: type(model))} =============================")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # Prompt: 'Give me a short introduction to large language model.', Generated text: ' A large language model, often referred to as a language model, is a type'
        print(
            f"============================= Prompt: {prompt!r}, Generated text: {generated_text!r} =============================")

    return outputs


if __name__ == "__main__":
    run_gpt()
