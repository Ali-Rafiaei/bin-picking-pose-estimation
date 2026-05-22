# Downloads trained model checkpoints from HuggingFace
# Requires: huggingface_hub - Can be installed via `pip install huggingface_hub`
# Usage: Run this script from the project's root directory. Default download
#  location is ./models/checkpoints/ which can be changed by modifying the local_dir
#  parameter in the snapshot_download function.

from huggingface_hub import snapshot_download
import os


snapshot_download(
    repo_id="Ali-Rafiaei/ibpc-pose-estimator",
    repo_type="model",
    local_dir=os.path.join(os.path.dirname(__file__), "checkpoints")
)

print("Checkpoints saved to models/checkpoints/")
print(f"Set model_dir to {os.path.abspath('models')} when launching the ROS2 node.")
