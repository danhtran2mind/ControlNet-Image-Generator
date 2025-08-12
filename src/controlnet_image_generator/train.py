import os
import sys
import subprocess

if __name__ == "__main__":
    # Path to train_controlnet.py
    controlnet_script = os.path.join(os.path.dirname(__file__), "..", 
                                     "third_party", "diffusers", "examples", "controlnet", "train_controlnet.py")
    
    # Construct the command: python + script path + command-line arguments
    command = [sys.executable, controlnet_script] + sys.argv[1:]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running train_controlnet.py: {e}", file=sys.stderr)
        sys.exit(e.returncode)