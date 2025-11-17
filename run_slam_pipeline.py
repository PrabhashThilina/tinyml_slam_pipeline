import subprocess
import time
import os

# Configuration
steps = [
    {
        "name": "Step 1: Key point identification",
        "cmd": "python3 run_superpoint.py processed",
        "workdir": "/home/thilina/workspace/Projects/TML/tinyml-slam-pipeline"
    },
    {
        "name": "Step 2: Depth Estimation",
        "cmd": "python3 run_depth_estimation.py",
        "workdir": "/home/thilina/workspace/Projects/TML/tinyml-slam-pipeline"
    },
    {
        "name": "Step 2: Segmentation",
        "cmd": "python3 run_segmentation.py",
        "workdir": "/home/thilina/workspace/Projects/TML/tinyml-slam-pipeline"
    },
    {
        "name": "Step 3: SLAM Backend (SuperPoint)",
        "cmd": "python3 run_slam_back_end.py",
        "workdir": "/home/thilina/workspace/Projects/TML/tinyml-slam-pipeline"
    }
]

# Sequential pipeline
def run_step(step):
    print(f"\n{'='*60}")
    print(f" {step['name']}")
    print(f"{'='*60}\n")

    start = time.time()
    try:
        result = subprocess.run(
            step["cmd"],
            shell=True,
            cwd=step["workdir"],
            check=True,
            text=True
        )
        print(f"\n {step['name']} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n ERROR in {step['name']}")
        print(e)
        return False

    duration = time.time() - start
    print(f" Duration: {duration:.2f} seconds\n")
    return True

# Main execution
if __name__ == "__main__":
    print("\n Starting Full TinyML SLAM Pipeline...\n")
    start_time = time.time()

    for step in steps:
        if not run_step(step):
            print("\n⚠️ Pipeline stopped due to an error.")
            break
    else:
        print("\n All stages completed successfully!")

    total_time = time.time() - start_time
    print(f"\n Total Pipeline Time: {total_time/60:.2f} minutes\n")
