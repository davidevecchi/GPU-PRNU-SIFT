import math
import statistics
import sys
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

sys.path.insert(1, 'CameraFingerprint')
import src.Functions as Fu
import src.Filter as Ft


class Method(Enum):
    ICIP, RAFT, NEW, RND = 0, 1, 2, 3
    # NO_INV = 3


class Mode(Enum):
    ALL, I0, GOP0 = 0, 1, 2


def get_tf_flags():
    FLAGS = tf.compat.v1.flags.FLAGS
    # dataset
    tf.compat.v1.flags.DEFINE_string('hypothesis', '1', 'hypothesis to check (1 or 0)')
    tf.compat.v1.flags.DEFINE_string('videos', 'vision/dataset/', 'path to videos')
    tf.compat.v1.flags.DEFINE_string('fingerprint', 'PRNU_fingerprints/', 'path to fingerprint')
    tf.compat.v1.flags.DEFINE_string('output', 'results/', 'path to output')
    tf.compat.v1.flags.DEFINE_string('gpu_dev', '/gpu:0', 'gpu device')
    tf.compat.v1.flags.DEFINE_string('method', 'ICIP', 'method')
    tf.compat.v1.flags.DEFINE_string('mode', 'ALL', 'mode')
    
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    return FLAGS


def get_noise_WT(resized, size_fing):
    noise = Ft.NoiseExtractFromImage(resized, sigma=2.)
    noise = Fu.WienerInDFT(noise, np.std(noise))
    W_T = tfa.image.transform(tf.convert_to_tensor(noise, dtype=tf.float32), [1, 0, 0, 0, 1, 0, 0, 0], 'BILINEAR', [size_fing[0], size_fing[1]])
    return noise, W_T


def get_matches(queryKeypoints, trainKeypoints, matches):
    med = statistics.median(
        [math.sqrt(
            (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 +
            (queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[1]) ** 2
        ) for match in matches]
    )
    matches = [match for match in matches if math.sqrt(
        (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 +
        (queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[1]) ** 2
    ) < med * 10]
    return matches


def _RemoveNeighborhood(X, x, ssize):
    # Remove a 2-D neighborhood around x=[x1,x2] from matrix X and output a 1-D vector Y
    # ssize     square neighborhood has size (ssize x ssize) square
    # noinspection PyUnusedLocal
    M, N = X.shape
    radius = (ssize - 1) / 2
    X = np.roll(X, [np.int(radius - x[0]), np.int(radius - x[1])], axis=[0, 1])
    Y = X[ssize:, :ssize]
    Y = Y.flatten()
    Y = np.concatenate([Y, X.flatten()[int(M * ssize):]], axis=0)
    return Y


def warp(noise, matrix, size_Fingeprint):
    list_Wrs = tf.expand_dims(
        tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise, dtype=tf.float32), axis=0), repeats=len(matrix), axis=0), axis=-1
    )
    batchW = tfa.image.transform(list_Wrs, matrix, 'BILINEAR', [size_Fingeprint[1], size_Fingeprint[2]])
    return list_Wrs, batchW
