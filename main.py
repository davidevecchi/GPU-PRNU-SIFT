from datetime import datetime
import logging
import os

import sys
import time

sys.path.insert(1, '/home/dnntrainer/.davv/prnu/CameraFingerprint')
from src.loader import CameraFingerprintLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

if __name__ == '__main__':
    loader = CameraFingerprintLoader()
    time0 = time.perf_counter()
    loader.start()
    time_diff = time.perf_counter() - time0
    dt = datetime.now().strftime(f'%Y-%m-%d %H:%M:%S')
    filename = 'performance.txt'
    with open(filename, 'r+') as f:
        lines = f.readlines()
        f.write(f'{dt}\t{loader.name}\t{time_diff}s\n')
        if loader.name[:2] in lines[-2]:
            f.write(f'{dt}\tTOT{loader.name[2:]}\t{time_diff}s\n')
