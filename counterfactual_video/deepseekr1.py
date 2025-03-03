import torch
from transformers import pipeline

# Initialize the text generation pipeline
pipe = pipeline(
    "text-generation",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    torch_dtype=torch.float16,
    device = 0 
)

# Properly format the input for the model
input_text = '''
Describe the text below by excluding the factors: age, gender, beard, baldness.

text:
The image shows a person with a shaved head, featuring a unique hairstyle with two prominent, 
symmetrical lines resembling eyebrows. The person has a neutral expression and is wearing a 
black shirt. The background is blurred, with a hint of a light source visible.
'''

# Run the pipeline with max_new_tokens to avoid truncation
output = pipe(input_text, max_new_tokens=100)  # Adjust token length as needed

# Print the generated text
print(output[0]["generated_text"])