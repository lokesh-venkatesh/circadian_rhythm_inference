import subprocess
import os
import sys
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="VAE Model Configuration")
    parser.add_argument('--input_size', type=int, default=64*24, help='Dimensions of the input data')
    parser.add_argument('--degree', type=int, default=3, help='Degree of fourier series for seasonal inputs')
    parser.add_argument('--latent_size', type=int, default=4*24, help='The hours associated with each latent variable')
    parser.add_argument('--latent_dim', type=int, default=None, help='Latent dimension')
    parser.add_argument('--latent_filter', type=int, default=10, help='Latent filter')
    parser.add_argument('--interim_filters', type=int, default=20, help='Interim filters')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--prior_dist_type', type=str, default='Seasonal', help='Prior Distribution type') # NOTE IMP!
    parser.add_argument('--activate_phase_shift', type=bool, default=False, help='Randomly shift input phases') # NOTE IMP!
    parser.add_argument('--gnrt_start', type=str, default='1970-01-01 00:00:00', help='Timestamp generation starts from')
    parser.add_argument('--gnrt_end', type=str, default='2020-12-31 16:00:00', help='Timestamp generation ends at')
    args, _ = parser.parse_known_args()
    return args

args = get_config()

# define constants
INPUT_SIZE = args.input_size
DEGREE = args.degree
LATENT_SIZE = args.latent_size
gnrt_start = args.gnrt_start
gnrt_end = args.gnrt_end

# Parameters
input_shape = INPUT_SIZE
latent_dim = args.latent_dim
latent_filter = args.latent_filter
interim_filters = args.interim_filters

# training hyperparameters
learning_rate = 0.001
epochs = args.epochs
batch_size = 32
prior_dist_type = args.prior_dist_type
activate_phase_shift = args.activate_phase_shift

os.makedirs('results', exist_ok=True)
results_directory = f'results/training_run_{prior_dist_type}_prior_{activate_phase_shift}_phase_shift_{epochs}_epochs'

if __name__=="__main__":
    scripts = ["data.py", 
               "train.py", 
               "generate.py", 
               "plot.py", 
               "latent.py", 
               "latent_space_megaplot.py",
               "organise.py"]
    all_logs = []
    for script in scripts:
        all_logs.append(f"\n--- Running {script} ---\n")
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        # Capture stdout and stderr
        result = subprocess.run(
            [sys.executable, script] + sys.argv[1:],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        all_logs.append(result.stdout)
        all_logs.append(f"--- Finished {script} ---\n")
    # Save all logs to logs.txt
    with open("logs.txt", "w", encoding="utf-8") as f:
        f.writelines(all_logs)
    # Also print to terminal
    print(''.join(all_logs))