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

def run_script_with_combinations(script_path, gsizes, num_agents_list):
    for gsize in gsizes:
        for num_agents in num_agents_list:
            print(f"Running for gsize={gsize}, num_agents={num_agents}...")
            result = run(
                ["python", script_path, str(gsize), str(num_agents)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Error with gsize={gsize}, num_agents={num_agents}: {result.stderr}")
                break
            print(result.stdout)

if __name__ == "__main__":
    # Get the notebook path, gsize values, and num_agents values from the user
    if len(sys.argv) < 4:
        print("Usage: python run_notebook_with_params.py <notebook_path> <gsize_values> <num_agents_values>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    gsizes = list(map(int, sys.argv[2].split(',')))
    num_agents_list = list(map(int, sys.argv[3].split(',')))

    if not os.path.exists(notebook_path):
        print(f"Error: The file '{notebook_path}' does not exist.")
        sys.exit(1)

    # Convert the notebook to a Python script
    script_path = convert_notebook_to_script(notebook_path)

    try:
        # Run the script for each combination of gsize and num_agents
        run_script_with_combinations(script_path, gsizes, num_agents_list)
    finally:
        # Clean up the generated Python script
        if os.path.exists(script_path):
            os.remove(script_path)


#python run_var.py qlearning_QCBRL_ma_decentralized_csv.ipynb "10,20,30" "2,4,6,8"