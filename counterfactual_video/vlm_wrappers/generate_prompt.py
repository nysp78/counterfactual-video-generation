import re

def generate_vlm_prompt(attributes, values, upstreams, all_attributes):
    excluded = set(all_attributes) - set(attributes)
    
    # If it's a downstream attribute, exclude its causal parents
    if "beard" in attributes or "bald" in attributes:
        excluded |= upstreams

    excluded_list = ', '.join(sorted(excluded))

    prompt = f"""
You are given an image of a person's face.

We assume a causal graph where: age cause beard and baldness, gender causes beard and baldness.
Your task is to generate a Stable Diffusion-style visual description of how the facial or head region would appear if the person had the attributes '{attributes}' with values '{values}'.
Describe only the visually grounded features directly related to this change. Use short, comma-separated phrases in the style of Stable Diffusion prompts — no full sentences or narrative.
Do not describe or modify any of the following: {excluded_list}, facial expression, emotion, hairstyle, background, clothing, lighting, or makeup — unless these are clearly visible and unchanged in the original image.
Return up to 6 comma-separated visual descriptors each reflecting a facial or head feature consistent with '{attributes} = {values}'
Your response must be a single line containing up to 6 comma-separated visual descriptors.

""".strip()

    return prompt

def gender_from_text(text):
    text = text.lower()
    if re.search(r'\b(he|man)\b', text):
        return "man"
    elif re.search(r'\b(she|woman)\b', text):
        return "woman"
    return None


def generate_vlm_prompt__(crf_prompt, target_interventions):
    
    bias_correction_prompt = ""
    
    crf_prompt = crf_prompt.lower()
    target_interventions = target_interventions.lower()
    crf_gender = gender_from_text(crf_prompt)

    if crf_gender == "woman" and ("beard" in target_interventions or "bald" in target_interventions) and ("no-beard" not in target_interventions) and ("no-bald" not in target_interventions):
        bias_correction_prompt = '''Take into account that may exist biases in cases we want to add beard/baldness on women. 
In such extreme cases derive prompts that break these biases and do not mention gender (e.g. individual with beard, a drag queen with beard, a bald drag queen individual with bald head etc).'''


    prompt = f'''
You are given an image of a person's face.
We are also given the counterfactual target prompt: {crf_prompt} interventions: {target_interventions}

{bias_correction_prompt}

Evaluate how well the generated image aligns with the specified counterfactual attributes from the target prompt.
Calculate an accuracy score based only on the attributes that were explicitly modified (interventions).
Identify and list any attributes from the interventions that failed to appear or were incorrectly rendered.
Suggest improvements to the counterfactual prompt to better achieve the intended attributes of the counterfactual prompt.

'''.strip()
    return prompt



#Evaluate how well the generated image aligns with the target attributes and the target prompt.
#Can you calculate a accuracy score on how well the given image aligns with the attributes mentioned in the counterfactual prompt?
#Return the score over the interventions only and the attributes that failed.