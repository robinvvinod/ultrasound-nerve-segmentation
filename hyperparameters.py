from losses import *

################################################################################
# Hyperparameters
################################################################################

# Leaky ReLU
alpha = 0.1

# Input Image Dimensions 
# (rows, cols, channels)
input_dimensions = (420,580,1)
# (rows, cols)
dimensions = (420,580)

# Dropout probability
dropout = 0.5

# Training parameters
num_initial_filters = 32
batchnorm = True
num_gpu = 1
batch_size = 2
steps_per_epoch = 8
learning_rate = 0.0001
loss = focal_tversky_loss
metrics = [dice_coef]
epochs = 2500

# Paths

checkpoint_path = 'checkpoint'
log_path = 'log.txt'
save_path = 'model'
