import logging
import os

import sys

sys.path.insert(1, '/home/dnntrainer/.davv/prnu/CameraFingerprint')
from src.loader import CameraFingerprintLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

if __name__ == '__main__':
    loader = CameraFingerprintLoader()
    loader.run_analysis()
