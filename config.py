import subprocess
import os
import sys
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="VAE Model Configuration")
    parser.add_argument('--input_size', type=int, default=64*24, help='Dimensions of the input data')
    parser.add_argument('--degree', type=int, default=1, help='Degree of fourier series for seasonal inputs')
    parser.add_argument('--latent_size', type=int, default=4*24, help='The hours associated with each latent variable')
    parser.add_argument('--latent_dim', type=int, default=None, help='Latent dimension')
    parser.add_argument('--latent_filter', type=int, default=10, help='Latent filter')
    parser.add_argument('--interim_filters', type=int, default=20, help='Interim filters')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    args, _ = parser.parse_known_args()
    return args

args = get_config()

# define constants
INPUT_SIZE = args.input_size
DEGREE = args.degree
LATENT_SIZE = args.latent_size

# Parameters
input_shape = INPUT_SIZE
latent_dim = args.latent_dim
latent_filter = args.latent_filter
interim_filters = args.interim_filters

# training hyperparameters
learning_rate = 0.001
epochs = args.epochs
batch_size = 32

if __name__=="__main__":
    scripts = ["data.py", "train.py", "generate.py", "plots.py"]
    scripts.extend(["latent.py","analysis_scripts/flatten_pca_day_of_year.py","analysis_scripts/flatten_pca_temp.py",
        "analysis_scripts/flatten_tsne_day_of_year.py","analysis_scripts/flatten_tsne_temp.py"])
    for script in scripts:
        print(f"\n--- Running {script} ---")
        # Suppress Python warnings and set environment variable to ignore warnings in subprocess
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        # Pass the same arguments to the subprocess
        subprocess.run([sys.executable, script] + sys.argv[1:], env=env)
        print(f"--- Finished {script} ---")