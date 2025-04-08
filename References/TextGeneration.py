from transformers import pipeline
import torch
import time

# Prompt
prompt = "We will be building a AI model for a Large Firm that"

# Models to test
models = [
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "facebook/opt-350m"
]

# Decoding strategies
decoding_params = [
    {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
    {"temperature": 0.7, "top_k": 40, "top_p": 0.9},
    {"temperature": 0.9, "top_k": 0, "top_p": 0.92}
]

# Automatically select GPU if available
device = 0 if torch.cuda.is_available() else -1
print(f"✅ Using {'GPU' if device == 0 else 'CPU'} for inference")

# Open file to save results
with open("text_generation_outputs.txt", "w") as file:
    file.write(f"Prompt: {prompt}\n\n")

    for model_name in models:
        print(f"\n🔄 Loading model: {model_name} ...")
        try:
            generator = pipeline("text-generation", model=model_name, device=device)
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            file.write(f"\n=== Model: {model_name} ===\n")
            file.write(f"Error: {e}\n\n")
            continue

        file.write(f"\n=== Model: {model_name} ===\n")
        file.write(f"Device: {'GPU' if device == 0 else 'CPU'}\n")

        for i, params in enumerate(decoding_params):
            print(f"⚙️  Generating with decoding strategy {i+1}: {params}")
            start = time.time()
            output = generator(prompt, max_length=50, **params)
            end = time.time()

            generated_text = output[0]['generated_text']

            file.write(f"\n--- Strategy {i+1} ---\n")
            file.write(f"Params: {params}\n")
            file.write(f"Time Taken: {end - start:.2f} sec\n")
            file.write("Generated Text:\n")
            file.write(generated_text + "\n")

    print("\n✅ All results saved to: text_generation_outputs.txt")
