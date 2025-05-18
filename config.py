# define constants
INPUT_SIZE = 64*24 # dimensions of the input data
DEGREE = 3 # degree of fourier series for seasonal inputs
LATENT_SIZE = 4*24 # the hours associated with each latent variable

# Parameters
input_shape = None #INPUT_SIZE
latent_dim = None #INPUT_SIZE//LATENT_SIZE
latent_filter = 10
interim_filters = 2*latent_filter

# training hyperparameters
learning_rate = 0.001
epochs = 100
batch_size = 32