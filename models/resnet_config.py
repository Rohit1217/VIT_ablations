# Model
NUM_LAYERS         = 30
PROJ_KERNEL        = 5
RESIDUAL_CHANNELS  = 30
RESIDUAL_KERNEL    = 5
STRIDE             = 1
PADDING            = 2
BIAS               = True
BATCH_NORM         = True
HIDDEN_DIM         = 512
NUM_CLASSES        = 10

# Training
DEVICE             = "cuda:3"
BATCH_SIZE         = 512
NUM_EPOCHS         = 200
WARMUP_EPOCHS      = 10
LR                 = 1e-3
WEIGHT_DECAY       = 0.1
LABEL_SMOOTHING    = 0.1
NUM_WORKERS        = 16
GRAD_CLIP          = 1.0

# Augmentation
MIXUP_ALPHA        = 0.2
CIFAR_MEAN         = (0.4914, 0.4822, 0.4465)
CIFAR_STD          = (0.2470, 0.2435, 0.2616)

# Checkpoint
CHECKPOINT_PATH    = "best_resnet.pt"
