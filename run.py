import subprocess

scripts = ["data.py", "train.py", "generate.py", "plots.py", "latent.py"]

for script in scripts:
    print(f"\n--- Running {script} ---")
    subprocess.run(["python", script])
    print(f"--- Finished {script} ---")