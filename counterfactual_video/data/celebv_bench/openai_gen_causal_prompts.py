import json
from openai import OpenAI
from tqdm import tqdm

api_key = "YOUR_OPENAI_KEY"

client = OpenAI(api_key=api_key)

# Load your source JSON
with open("source_prompts.json", "r") as f:
    source_data = json.load(f)

system_prompt = """
You are given a causal DAG with 4 variables: age, gender, beard, baldness.

Causal relationships:
- age affects beard
- age affects bald
- gender affects beard
- gender affects bald

Pay attention to domain knowledge and causal relationships:
1. Old men are more likely to have a beard and be bald, than young men
2. Men are more likely to have a beard or be bald than women
3. When transforming a man into a woman remove the beard or baldness from the counterfactual prompt
4. When transforming an old or young woman into an old or young man put beard (if the man is old put also baldness) in the counterfactual prompt
5. When transforming an old man with a beard or baldness into a young man remove beard or baldness from the counterfactual prompt
6. When transforming a young man to old put beard or baldness in the counterfactual prompt


Pay attention to Rules
Rules:
intervention on age:
if transform to young:
    If gender = man:
        remove beard or baldness from the counterfactual prompt
    else if gender = woman:
        do nothing

if transform to old:
     If gender = man:
        put beard and baldness in the counterfactual prompt
    else if gender = woman:
        do nothing

intervention on gender:
if transform to man:
    if age = old
      put beard and baldness in the counterfactual prompt
    
    else if age = young:
        put only beard in the counterfactual prompt

else if transform to woman:
    remove beard and baldness from the counterfactual prompt

interventions on beard , baldness:
if beard = yes or baldness = yes -> beard or baldness:
    put beard or baldness in the counterfactual prompt
else:
    remove beard or baldness from the counterfactual prompt



Task:
Given a factual description of a person, generate counterfactuals by intervening on based on the domain knowledge:
- age
- gender
- beard
- bald

Pay attention to:

a)If a non-intevened attribute that is independent from the intervened attribute , exists in the source (factual) prompt, also retain this non-intervene attribute in the counterfactual prompt
    i)  Example1: He has beard, he is old -> (after bald intervention) He has beard, he is old, he is bald (keep beard)
    ii) Example2: He is young, he is bald -> (afer beard intervention) He is young, he is bald, he has a beard (keep bald)

b)If an intervention happens on upstream variables (age, gender) adjust according to the Rules and the Domain Knowledge the downstream depedent variables (beard, bald) in the counterfactual prompt
    i) Example: A man is old, has a beard -> (after age intervention) a man is young (remove also beard)

c)If an intervention removes an attribute (e.g. beard or baldness) do not mention the subtracted attribute in the counterfactual prompts

d) Given a factual prompt that describes a person (e.g., He is young, he has a beard),
generate 4 counterfactual prompts by intervening on each variable (age, gender,beard, bald) while respecting the causal relationships.

e) Do not inlude any negations when an attribute has to be removed (e.g. no beard , no bald), just remove completely the reference from the prompt

Output a JSON object with keys: age, gender, beard, bald.

Examples:
---
Factual:
He is young
Counterfactuals:
age: He is old, he has a beard, he is bald
gender: She is young
beard: He is young, he has a beard
bald: He is young, he is bald
---
Factual:
He is young, he has a beard
Counterfactuals:
age: He is old, he has a beard, he is bald
gender: She is young
beard: He is young
bald: He is young, he has a beard, he is bald
---
Factual:
He is old, he is bald
Counterfactuals:
age: He is young
gender: She is old
beard: He is old, he has a beard, he is bald
bald: He is old
---
Factual:
She is old
Counterfactuals:
age: She is young
gender: He is old, he has a beard, he is bald
beard: She is old, she has a beard
bald: She is old, she is bald
"""

results = {}

for uid, factual_prompt in tqdm(source_data.items()):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Factual:\n{factual_prompt}\n\nGenerate counterfactuals as JSON."}
        ]
    )

    try:
        cf = json.loads(completion.choices[0].message.content)
    except:
        cf = {"error": "Invalid JSON", "raw": completion.choices[0].message.content}

    results[uid] = {
        "factual": factual_prompt,
        "counterfactual": cf
    }

# Save to file
with open("counterfactuals__v2.json", "w") as f:
    json.dump(results, f, indent=4)

print("counterfactuals.json created")
