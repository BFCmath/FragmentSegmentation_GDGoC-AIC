# --- Application Settings ---
ALLOWED_CONTENT_TYPES = ['image/jpeg', 'image/png']
DEV_MODE = True  # Set to True for development, overridden by ENV var
PORT = 3000

# --- Model Paths ---
RGB_WEIGHT_PATH_TEMPLATE = 'weights/yolo_rgb_{}.onnx'
RGBD_WEIGHT_PATH_TEMPLATE = 'weights/yolo_rgbd_{}.onnx'
DEPTH_MODEL_PATH_TEMPLATE = 'weights/depth_anything_v2_{}.pth' 

# --- Model Parameters ---
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.7
YOLO_RETINA_MASKS = True
YOLO_SHOW_LABELS = False
YOLO_SHOW_CONF = False
YOLO_SHOW_BOXES = False

# Shared Image Dimensions
MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 512

# --- Device Settings ---
DEVICE = 'cpu'

# --- Depth Model Specific Configs ---
# Options: 'vits', 'vitb', 'vitl', 'vitg'
DEPTH_MODEL_TYPE = 'vits' # Default
DEPTH_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'api.log' 