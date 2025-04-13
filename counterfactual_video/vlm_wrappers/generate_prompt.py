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