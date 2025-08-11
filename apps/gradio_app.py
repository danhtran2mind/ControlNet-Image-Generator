import os
import subprocess



def run_setup_script():
    setup_script = os.path.join(os.path.dirname(__file__),
                                "gradio_app", "setup_scripts.py")
    try:
        result = subprocess.run(["python", setup_script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Setup script failed with error: {e.stderr}")
        return f"Setup script failed: {e.stderr}"
    
def create_gui():
    





if __name__ == "__main__":
    run_setup_script()
    demo = create_gui()
    demo.launch(debug=True)