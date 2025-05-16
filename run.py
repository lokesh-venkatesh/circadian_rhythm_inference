import subprocess

scripts = ["process_data.py", "vae.py", "plots.py"]
log_file = "logs.txt"

with open(log_file, "w") as log:
    for script in scripts:
        log.write(f"\n--- Running {script} ---\n")
        result = subprocess.run(
            ["python", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        log.write(result.stdout)
        log.write(f"\n--- Finished {script} ---\n")