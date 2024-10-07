import os
import sys
import nbformat
from nbconvert import PythonExporter
from subprocess import run

def convert_notebook_to_script(notebook_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Convert the notebook to a Python script
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(notebook)

    # Create a temporary script file
    script_name = os.path.splitext(os.path.basename(notebook_path))[0] + ".py"
    script_path = os.path.join(os.path.dirname(notebook_path), script_name)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    
    return script_path

def run_script_n_times(script_path, n):
    for i in range(n):
        print(f"Running iteration {i + 1}...")
        result = run(["python", script_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error on iteration {i + 1}: {result.stderr}")
            break
        print(result.stdout)

if __name__ == "__main__":
    # Get the notebook path and number of iterations from the user
    if len(sys.argv) != 3:
        print("Usage: python run_notebook_n_times.py <notebook_path> <n>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    n = int(sys.argv[2])

    if not os.path.exists(notebook_path):
        print(f"Error: The file '{notebook_path}' does not exist.")
        sys.exit(1)

    # Convert the notebook to a Python script
    script_path = convert_notebook_to_script(notebook_path)

    try:
        # Run the script n times
        run_script_n_times(script_path, n)
    finally:
        # Clean up the generated Python script
        if os.path.exists(script_path):
            os.remove(script_path)


#to run script in cmd:
#python run.py qlearning_QCBRL_ma_decentralized_csv.ipynb 5


