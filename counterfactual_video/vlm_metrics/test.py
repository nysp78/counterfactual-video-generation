import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

p = '''
You are a prompt generation assistant for Stable Diffusion.

Your task is to generate a rich, vivid, and visually descriptive Stable Diffusion prompt that focuses only on the given attribute and value — in this case, describing the **facial or head region** as it relates to the attribute.

Use realistic and visually grounded language that directly expresses the attribute and target value.  
Use comma-separated descriptors in the Stable Diffusion style. The prompt must contain no more than 4 tokens (phrases separated by commas).
Do not include any information related to the following attributes: gender, beard, baldness.  


Attribute: age  
Target value: young
When you generate SD prompt for downstream causal variables(beard,baldness) exclude infromation related to the upsteams variables(age, gender)
Your output must include:

Generated SD prompt: [your_prompt_here]

'''
prompt_2 = '''You are given an image of a person's face. 
Generate a Stable Diffusion-style visual description of how the facial or head region would appear if the person was 'young'. Use expressive, realistic descriptors of youth-related facial features only. Do not describe or modify any other visual elements such as gender, facial expression, hairstyle, background, clothing, lighting, or makeup unless they are unchanged and clearly visible in the original image. 
Your output must consist of exactly 4 comma-separated phrases, each describing a visually grounded change related to the attribute 'young'.'''

 #single image conversation example
# Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
def generate_vlm_prompt(attribute, value, upstreams, all_attributes):
    excluded = set(all_attributes) - {attribute}
    
    # If it's a downstream attribute, exclude upstreams too
    if attribute in {"beard", "bald/hair"}:
        excluded |= upstreams

    excluded_list = ', '.join(sorted(excluded))

    prompt = f"""
You are given an image of a person's face.
Your task is to describe how the facial or head region would visually appear if the person had the attribute '{attribute}' set to '{value}'.
Use only short, visually grounded phrases. Do not write full sentences or explanations.
Your output must follow the style of Stable Diffusion prompts: concise, comma-separated visual tokens describing only the facial or head features affected by this attribute.
Do not describe or modify any of the following: {excluded_list}, facial expression, emotion, hairstyle, background, accessories, clothing, lighting, or makeup — unless they are clearly visible and unchanged in the original image.
Return exactly 6 comma-separated visual descriptors**, each representing a visual trait consistent with '{attribute} = {value}'.
""".strip()

    return prompt

all_attributes = ["age", "gender", "beard", "bald/hair"]
upstream_attributes = {"age", "gender"}

prompt2 = generate_vlm_prompt("age", "young", upstreams=upstream_attributes, all_attributes=all_attributes)
print(prompt2)


conversation = [
    {
        "role": "<|User|>",
        "content": f"<image>{prompt2}",
        "images": ["/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_1.5/explicit/interventions/age/1MO3gP8vxoE_3_0/youthful, vibrant, glowing complexion, smooth skin, energetic/vae_recon/00000.png"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

with torch.no_grad():
# run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1000,
        do_sample=True,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    print(f"{prepare_inputs['sft_format'][0]}", answer)