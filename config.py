import subprocess
import os
import sys

# define constants
INPUT_SIZE = 64*24 # dimensions of the input data
DEGREE = 3 # degree of fourier series for seasonal inputs
LATENT_SIZE = 4*24 # the hours associated with each latent variable

# Parameters
input_shape = INPUT_SIZE # None
latent_dim = None # INPUT_SIZE//LATENT_SIZE # None
latent_filter = 10
interim_filters = 40 # 2*latent_filter

# training hyperparameters
learning_rate = 0.001
epochs = 100
batch_size = 32

if __name__=="__main__":

    scripts = ["data.py", "train.py", "generate.py", "plots.py", "latent.py"]
    for script in scripts:
        print(f"\n--- Running {script} ---")
        # Suppress Python warnings and set environment variable to ignore warnings in subprocess
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        subprocess.run([sys.executable, script], env=env)
        print(f"--- Finished {script} ---")