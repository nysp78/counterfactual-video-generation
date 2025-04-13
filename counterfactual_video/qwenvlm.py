from transformers import pipeline

#messages = [
#    {"role": "user", "content": "Who are you?"},
#]
#pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-3B-Instruct")
#pipe(messages)

from transformers import pipeline
from PIL import Image

# Load the model
pipe = pipeline("image-to-text", model="Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto")

# Load an image
image = Image.open("/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_2.5/explicit/interventions/bald/-_B4fiuWwmo_0_1/smooth bald scalp, reflective skin, hairless head, realistic texture/vae_recon/00000.png").convert("RGB")
#print(type(image))
#path = "/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_2.5/explicit/interventions/bald/-_B4fiuWwmo_0_1/smooth bald scalp, reflective skin, hairless head, realistic texture/vae_recon/00000.png"
# Ask a question about the image
prompt = "Describe this image."

# Run the pipeline
output = pipe({
    "image": image,
    "prompt": prompt
})
print(output[0]["generated_text"])

#image_path =  "/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_2.5/explicit/interventions/bald/-_B4fiuWwmo_0_1/smooth bald scalp, reflective skin, hairless head, realistic texture/vae_recon/00000.png"


