import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Resize, ToPILImage, Compose
from collections import defaultdict
import glob
from utils import extract_nth_frame, LlavaNext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_path", type=str, default="/path/to/generated_videos")
    parser.add_argument("--method", choices=["tuneavideo", "tokenflow", "flatten"], default="flatten")
    parser.add_argument("--questions_path", type=str, default='../data/celebv_bench/questions_explicit.json')
    opt = parser.parse_args()

    # Load questions
    with open(opt.questions_path, "r") as f:
        multiple_choice_questions = json.load(f)

    intervention_type = "explicit"
    model = LlavaNext(model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda")
    
    transform = Compose([ToPILImage(), Resize((512, 512))])

    attributes = ["age", "gender", "beard", "bald"]
    interventions = ["age", "gender", "beard", "bald"]

    # Initialize accuracy matrix: attr × intervention
    accuracy_matrix = {
        attr: {interv: {"correct": 0, "total": 0} for interv in interventions}
        for attr in attributes
    }

    for video_id, questions in tqdm(multiple_choice_questions.items()):
        for intervention_attr, q_data in questions.items():
            crf_prompt = q_data["prompt"]
            base_path = f"{opt.outputs_path}/{opt.intervention_type}/interventions/{intervention_attr}/{video_id}/{crf_prompt}"

            # Select appropriate video path
            if opt.method == "flatten":
                pattern = f"{base_path}/*_ugly, blurry, low res, unrealistic, unaesthetic_7.5.mp4"
                video_path = glob.glob(pattern)[0]
            elif opt.method == "tokenflow":
                video_path = f"{base_path}/tokenflow_PnP_fps_20.mp4"
            elif opt.method == "tuneavideo":
                video_path = f"{base_path}/edited_fps20.gif"
            else:
                continue

            try:
                frame = extract_nth_frame(video_path, n=7)
                frame = transform(frame)
            except Exception as e:
                print(f"Failed to extract frame for {video_id}: {e}")
                continue

            for q in q_data["questions"]:
                question_text = q["question"]
                correct_answer = q["correctAnswer"].lower().strip().replace(".", "")
                ans_a = q["options"]["A"]
                ans_b = q["options"]["B"]

                # Identify attribute from question
                if "age" in question_text:
                    attr = "age"
                elif "gender" in question_text:
                    attr = "gender"
                elif "beard" in question_text:
                    attr = "beard"
                elif "bald" in question_text:
                    attr = "bald"
                else:
                    continue

                multi_choice_que = f"{question_text}, select from: [{ans_a}, {ans_b}], return only 1 word"
                pred_answer = model.generate(frame, multi_choice_que, do_sample=False)
                #print(pred_answer)
                pred_answer = pred_answer.split("[/INST]")[1].lower().strip().replace(".", "")
                is_correct = (pred_answer == correct_answer)
                print(pred_answer, correct_answer)
                if attr == "age" and intervention_attr == "gender":
                    print(video_id, is_correct, pred_answer, correct_answer)
                accuracy_matrix[attr][intervention_attr]["correct"] += int(is_correct)
                accuracy_matrix[attr][intervention_attr]["total"] += 1
                #print(accuracy_matrix[attr][intervention_attr]["total"])

    # Compute final accuracies
    print("Accuracy Matrix (attribute x intervention):")
    for attr in attributes:
        row = {}
        for interv in interventions:
            data = accuracy_matrix[attr][interv]
            acc = round(data["correct"] / data["total"], 3) if data["total"] > 0 else None
            row[interv] = acc
        print(f"{attr}_acc: {row}")