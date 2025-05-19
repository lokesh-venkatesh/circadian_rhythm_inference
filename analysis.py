import subprocess
import os
import sys
import argparse

if __name__=="__main__":

    scripts = ["analysis_scripts/flatten_pca_day_of_year.py", 
               "analysis_scripts/flatten_pca_temp.py", 
               "analysis_scripts/flatten_tsne_day_of_year.py", 
               "analysis_scripts/flatten_tsne_temp.py"]
    for script in scripts:
        print(f"\n--- Running {script} ---")
        # Suppress Python warnings and set environment variable to ignore warnings in subprocess
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        # Pass the same arguments to the subprocess
        subprocess.run([sys.executable, script] + sys.argv[1:], env=env)
        print(f"--- Finished {script} ---")