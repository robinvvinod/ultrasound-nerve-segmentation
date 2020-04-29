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
batch_size = 4
learning_rate = 0.00001
loss = focal_tversky_loss
metrics = [dice_coef]
epochs = 4000

# Paths

checkpoint_path = '/kaggle/working/checkpoint'
log_path = '/kaggle/working/log.txt'
save_path = '/kaggle/working/model'
