# VGG_FACE = r'/home/<user>/Documents/NeuralNetworkModels/vgg_face_dag.pth'
VGG_FACE = "./vgg_face_dag.pth"
LOG_DIR = "./log/"
MODELS_DIR = "./models"
GENERATED_DIR = "./generated_img"

# Dataset parameters
FEATURES_DPI = 100
K = 8

# Training hyperparameters
IMAGE_SIZE = 256  # 224
BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE_E_G = 5e-5
LEARNING_RATE_D = 2e-4
LOSS_VGG_FACE_WEIGHT = 2e-3
LOSS_VGG19_WEIGHT = 1e-2
LOSS_MCH_WEIGHT = 8e1
LOSS_FM_WEIGHT = 1e1
FEED_FORWARD = True
SUBSET_SIZE = 140000

# Model Parameters
E_VECTOR_LENGTH = 512
HIDDEN_LAYERS_P = 4096
