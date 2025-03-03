import json
import torch
import numpy as np
from tqdm import tqdm
import cv2
from torchmetrics.text import BLEUScore
from torchvision.transforms import Resize, ToPILImage, Compose
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    ret, frame = cap.read()
    cap.release()
    return frame

if __name__ == '__main__':
    # Load configuration
    with open('data/celebv_bench/counterfactual_explicit.json', "r") as f:
        crf_prompts = json.load(f)

    # Assign DeepSeek-R1 (LLM) to GPU 0
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Keep tokenizer on CPU
    deepseekr1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")

    # Assign DeepSeek-VL2 (VLM) to GPU 1
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer_vl = vl_chat_processor.tokenizer  # Keep tokenizer on CPU
    vl_gpt = DeepseekVLV2ForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:1")

    # Initialize BLEU scores
    bleu_scores = []
    num = 0
    transform = Compose([ToPILImage(), Resize((512, 512))])

    for video_id, prompts in tqdm(crf_prompts.items()):
        print("Evaluating video:", video_id)
        num += 1

        factual_frame = extract_first_frame(video_path=f"data/celebv_bench/videos/{video_id}.mp4")
        factual_frame = transform(factual_frame)

        for attr in prompts["counterfactual"].keys():
            vl_gpt.eval()
            crf_prompt = prompts["counterfactual"][attr]

            base_path = f"outputs/tuneavideo-results_cfg_scale_4.5/explicit/interventions/{attr}/{video_id}/{crf_prompt}"
            counterfactual_frame = extract_first_frame(video_path=base_path + "/edited_fps20.gif")
            counterfactual_frame = transform(counterfactual_frame)

            frames = [factual_frame, counterfactual_frame]

            conversations = [
                [{"role": "<|User|>", "content": "<image>Can you describe this image in detail?"},
                 {"role": "<|Assistant|>", "content": ""}],
                [{"role": "<|User|>", "content": "<image>Can you describe this image in detail?"},
                 {"role": "<|Assistant|>", "content": ""}]
            ]

            answers = []
            for i, conversation in enumerate(conversations):
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=[frames[i]],
                    force_batchify=True
                ).to("cuda:1")  # ✅ Send input tensors to the correct device

                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # Generate text descriptions with VL model
                outputs = vl_gpt.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer_vl.eos_token_id,
                    bos_token_id=tokenizer_vl.bos_token_id,
                    eos_token_id=tokenizer_vl.eos_token_id,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True
                )
                answer = tokenizer_vl.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                answers.append(answer)

            # **Process with DeepSeek-R1 on GPU 0**
            factual_description = f"Describe the text below by excluding the factors: age, gender, beard, baldness.\n text:\n{answers[0]}"
            counterfactual_description = f"Describe the text below by excluding the factors: age, gender, beard, baldness.\n text:\n{answers[1]}"

            # ✅ Tokenizer remains on CPU, but move tensors to GPU 0
            factual_inputs = tokenizer(factual_description, return_tensors="pt").to("cuda:0")
            counterfactual_inputs = tokenizer(counterfactual_description, return_tensors="pt").to("cuda:0")

            # Generate filtered descriptions using DeepSeek-R1 on GPU 0
            factual_filtered = deepseekr1.generate(**factual_inputs, max_new_tokens=100)
            counterfactual_filtered = deepseekr1.generate(**counterfactual_inputs, max_new_tokens=100)

            # Decode results
            pred = [tokenizer.decode(counterfactual_filtered[0], skip_special_tokens=True)]
            target = [[tokenizer.decode(factual_filtered[0], skip_special_tokens=True)]]

            # Compute BLEU score
            bleu_score = BLEUScore()(pred, target)
            bleu_scores.append(bleu_score)
            print("Video ID:", video_id, "BLEU Score:", bleu_score)

        if num > 11:
            break

    # Final BLEU score
    bleu_scores = np.array(bleu_scores)
    print("Final BLEU Score:", np.mean(bleu_scores))
