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
learning_rate = 0.00001
loss = tversky_loss
metrics = [dice_coef]
epochs = 20

# Paths

checkpoint_path = '/Users/RobinVinod/Documents/Coding/ML/ultrasound_nerve_segmentation/models/checkpoint.h5'
log_path = '/Users/RobinVinod/Documents/Coding/ML/ultrasound_nerve_segmentation/models/log.txt'
save_path = '/Users/RobinVinod/Documents/Coding/ML/ultrasound_nerve_segmentation/models/model.h5'
