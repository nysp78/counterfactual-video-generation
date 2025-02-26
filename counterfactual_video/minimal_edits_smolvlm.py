import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # Ensure model uses bfloat16
    _attn_implementation="flash_attention_2"
)

# ✅ Move model to GPU
model.to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs/tokenflow-results_cfg_scale_4.5/explicit/interventions/age/-_B4fiuWwmo_0_1/He is old, he has beard, he is bald./input_fps20.mp4"},
            {"type": "text", "text": "Describe in detail the image (person characteristics, background, style etc.)"}
        ]
    },
]

# Process inputs
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# ✅ Convert `pixel_values` to `bfloat16` to match model dtype
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

# ✅ Convert `input_ids` to `torch.int64` (LongTensor)
if "input_ids" in inputs:
    inputs["input_ids"] = inputs["input_ids"].to(torch.int64)

print(inputs["input_ids"].shape)
print(model.config)
# ✅ Move inputs to GPU
inputs = {key: value.to("cuda") for key, value in inputs.items()}

# Generate output
generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)

generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])