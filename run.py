import subprocess

scripts = ["process_data.py", "vae.py", "plots.py", "latent_space.py"]
log_file = "logs.txt"

with open(log_file, "w") as log:
    for script in scripts:
        log.write(f"\n--- Running {script} ---\n")
        log.flush()
        process = subprocess.Popen(
            ["python", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            print(line, end='')      # Print to terminal
            log.write(line)          # Write to log file
            log.flush()
        process.wait()
        log.write(f"\n--- Finished {script} ---\n")
        log.flush()