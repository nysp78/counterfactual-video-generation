import re

def gender_from_text(text):
    text = text.lower()
    if re.search(r'\b(he|man)\b', text):
        return "man"
    elif re.search(r'\b(she|woman)\b', text):
        return "woman"
    return None


def generate_vlm_prompt__(intervened_attr, crf_prompt, target_interventions, causal_dec=True):
    
    causal_decoupling_prompt = ""
    
    crf_prompt = crf_prompt.lower()
    target_interventions = target_interventions.lower()
    #crf_gender = gender_from_text(crf_prompt)

    #graph_mutilation
    if causal_dec:
        if intervened_attr == "beard" or intervened_attr == "bald":
            causal_decoupling_prompt = '''If either beard or bald appears in target interventions, do not include references to the values of age or gender (neutrilize gender (e.g. an individual or a person))'''


    prompt = f'''
You are given an image of a person's face.
You are also given the counterfactual target prompt: {crf_prompt} interventions: {target_interventions}

{causal_decoupling_prompt}

Evaluate how well the generated image aligns with the specified counterfactual attributes from the target prompt.
Calculate an accuracy score based only on the attributes that were explicitly modified (interventions).
Do not describe or modify any other visual elements such as expression, hairstyle, background, clothing, lighting.
Identify and list any attributes from the interventions that failed to appear or were incorrectly rendered.
Suggest improvements to the counterfactual prompt to better achieve the intended attributes of the counterfactual prompt.

'''.strip()
    return prompt




def generate_rephrasing_prompt(crf_prompt):

    prompt = f'''Paraphrase the target prompt:{crf_prompt}, use expressive and vibrant language.'''
    return prompt    