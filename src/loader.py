import glob
import os.path

from src import utils
from src.fingerprint import FingerprintAnalyzer
import roc_curve
from src.utils import Method


class CameraFingerprintLoader:
    def __init__(self):
        FLAGS = utils.get_tf_flags()
        self.hypothesis = int(FLAGS.hypothesis)
        self.test_set = FLAGS.videos
        self.gpu_dev = FLAGS.gpu_dev
        self.method = utils.Method[FLAGS.method]
        self.mode = utils.Mode[FLAGS.mode]
        self.name = f'H{self.hypothesis}_{self.method.name}_{self.mode.name}'
        self.output_path = os.path.join(FLAGS.output, self.name)
        self.fingerprint_paths = sorted(glob.glob(os.path.join(FLAGS.fingerprint, '*')))
        if self.hypothesis == 1 and self.method == Method.NEW:
            _, self.h0_percentile, self.h0_max = roc_curve.load_videos_pce(self.output_path.replace('H1', 'H0'), perc=95)  # , devices = None)  # FIXME !!!
            print(self.h0_percentile, self.h0_max)
            
    def start(self):
        print(self.output_path)
        for fingerprint_path in self.fingerprint_paths:
            # device = fingerprint_path[-7:-4]  # FIXME!!!!!!!!!
            # if device in ['D25']:  # FIXME!!!!!!!!!
            faa = FingerprintAnalyzer(self, fingerprint_path, self.method, self.mode)
            faa.run_analysis()
