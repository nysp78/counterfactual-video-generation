import os
import yaml
import torch
import subprocess
from pathlib import Path
from methods.tokenflow.run_tokenflow_pnp import TokenFlow
from methods.tokenflow.util import seed_everything
import os
import yaml
import json
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices =["tuneavideo", "tokenflow"], default="tokenflow")
    parser.add_argument('--base_config_path', type=str, default='methods/tokenflow/configs/config_pnp.yaml')
    parser.add_argument('--counterfactuals_config_path', type=str, default='data/celebv_bench/counterfactual_explicit.json')

    opt = parser.parse_args()

    with open(opt.base_config_path, "r") as f:
        config = yaml.safe_load(f)
        base_path = config["output_path"]

    with open(opt.counterfactuals_config_path, "r") as f:
        edited_prompts = json.load(f)

    seed_everything(config["seed"])

    
    for video_id, prompts in tqdm(edited_prompts.items()):
        config["data_path"] = f"data/celebv_bench/frames/{video_id}"
        config["video"][video_id] = {
                         # Assumes your videos are in "data/video_id"
                       "prompt_variants": {
                                             "factual": prompts["factual"],
                                             "counterfactual": prompts["counterfactual"]
                                         }
                   }

        for attr in prompts["counterfactual"].keys():
            config["output_path"] = os.path.join(config["output_path"], attr,
                                             video_id, config["video"][video_id]["prompt_variants"]["counterfactual"][attr])
            os.makedirs(config["output_path"], exist_ok=True)
            config["prompt"] = config["video"][video_id]["prompt_variants"]["counterfactual"][attr]
            assert os.path.exists(config["data_path"]), "Data path does not exist"
            print(config)

            if opt.method == "tokenflow":
                pipeline = TokenFlow(config)
                pipeline.edit_video()

            config["output_path"] = base_path #init base config output