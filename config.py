# VGG_FACE = r'/home/luccas/Documents/NeuralNetworkModels/vgg_face_dag.pth'
VGG_FACE = r'/home/lucas_fudstorm_com/models/vgg_face_dag.pth'
LOG_DIR = r'logs'
MODELS_DIR = r'models'
GENERATED_DIR = r'generated_img'

# Dataset parameters
FEATURES_DPI = 100
K = 8

# Training hyperparameters
IMAGE_SIZE = 224
EPOCHS = 100
LEARNING_RATE_E_G = 5e-5
LEARNING_RATE_D = 2e-4
LOSS_VGG_FACE_WEIGHT = 2e-3
LOSS_VGG19_WEIGHT = 1e-2
LOSS_MCH_WEIGHT = 8e1
LOSS_FM_WEIGHT = 1e1
