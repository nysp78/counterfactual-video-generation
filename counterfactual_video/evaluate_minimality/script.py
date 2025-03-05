import subprocess

def run_commands(commands):
    processes = []
    
    for command in commands:
        print(f"Running command: {command}")
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
    
    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":
    description_paths = [
        "raw_descriptions_tokenflow_explicit.json",
        "raw_descriptions_tuneavideo_explicit.json",
        "raw_descriptions_tokenflow_implicit.json",
        "raw_descriptions_tuneavideo_implicit.json",
        "raw_descriptions_tokenflow_breaking_causal.json",
        "raw_descriptions_tuneavideo_breaking_causal.json"
    ]
    
    commands = [
        f"CUDA_VISIBLE_DEVICES={i} python filter_description.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --description_path {desc} > {desc.split('_')[2]}_{desc.split('_')[3]}.txt"
        for i, desc in enumerate(description_paths)
    ]
    
    run_commands(commands)
