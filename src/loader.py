import glob
import os.path

from src import utils
from src.fingerprint import FingerprintAnalyzer


class CameraFingerprintLoader:
    def __init__(self):
        FLAGS = utils.get_tf_flags()
        self.hypothesis = int(FLAGS.hypothesis)
        self.test_set = FLAGS.videos
        self.gpu_dev = FLAGS.gpu_dev
        self.output_path = FLAGS.output
        self.fingerprint_paths = sorted(glob.glob(os.path.join(FLAGS.fingerprint, '*')))
    
    def run_analysis(self):
        for fingerprint_path in self.fingerprint_paths:
            fingerprint = FingerprintAnalyzer(self, fingerprint_path)
            fingerprint.run_analysis()
