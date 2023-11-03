import math

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src import utils


def crosscorr_Fingeprint_GPU(batchW, TA, norm2, sizebatch_K):
    meanW_batch = (tf.repeat(
        tf.repeat(
            (tf.expand_dims(
                tf.expand_dims(
                    tf.reduce_mean(
                        batchW, axis=[1, 2]
                    ), axis=2
                ), axis=3
            )),
            repeats=[sizebatch_K[1]], axis=1
        ), repeats=[sizebatch_K[2]], axis=2
    ))
    batchW = batchW - meanW_batch
    normalizator = tf.math.sqrt(tf.reduce_sum(tf.math.pow(batchW, 2)) * norm2)
    FA = tf.signal.fft2d(tf.cast(tf.squeeze(batchW, axis=3), tf.complex64))
    AC = tf.multiply(FA, tf.repeat(tf.cast(TA, dtype=tf.complex64), axis=0, repeats=len(batchW.numpy())))
    return tf.math.real(tf.signal.ifft2d(AC)) / normalizator


def parallel_PCE(CXC, idx, ranges, squaresize=11):
    out = np.zeros(idx)
    for i in range(0, idx):
        shift_range = ranges[i]
        Out = dict(PCE=[], pvalue=[], PeakLocation=[], peakheight=[], P_FA=[], log10P_FA=[])
        C = CXC[i]
        Cinrange = C[-1 - shift_range[0]:, -1 - shift_range[1]:]
        # noinspection PyUnusedLocal
        max_cc, imax = np.max(Cinrange.flatten()), np.argmax(Cinrange.flatten())
        ypeak, xpeak = np.unravel_index(imax, Cinrange.shape)[0], np.unravel_index(imax, Cinrange.shape)[1]
        Out['peakheight'] = Cinrange[ypeak, xpeak]
        del Cinrange
        Out['PeakLocation'] = [shift_range[0] - ypeak, shift_range[1] - xpeak]
        C_without_peak = utils._RemoveNeighborhood(C, np.array(C.shape) - Out['PeakLocation'], squaresize)
        # signed PCE, peak-to-correlation energy
        PCE_energy = np.mean(C_without_peak * C_without_peak)
        out[i] = (Out['peakheight'] ** 2) / PCE_energy * np.sign(Out['peakheight'])
    return out


def calibration_GPU(hom, noise, centerrot, centerres, step, TA, norm2, size_Fingeprint, matrix_off, bestpce=0):
    # Usa tf function per stima in parallelo di cross corr e PCE
    rotation = 0
    scaling = 0
    modifiedcheck = False
    # rotation estimation
    matrix, thetas = rotation_matrix_estimator(hom, noise.shape, centerrot, centerres, step)
    list_Wrs, batchW = utils.warp(noise, matrix, size_Fingeprint)
    ranges = np.repeat(
        [[size_Fingeprint[1] - noise.shape[0], size_Fingeprint[2] - noise.shape[1]]], repeats=100, axis=0
    )
    
    if len(matrix) >= 50:
        batches = 4
        batch_size = 25
        PCE_arr = list([None] * batches)
        for i in range(batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            
            XC = crosscorr_Fingeprint_GPU(batchW[start_idx:end_idx], TA, norm2, size_Fingeprint)
            PCE_arr[i] = parallel_PCE(XC.numpy(), batch_size, ranges[start_idx:end_idx])
            
            # pce_max = np.max(PCE_arr)
            # if not math.isnan(pce_max) and pce_max > bestpce:
            #     bestpce = pce_max
            #     idx = np.argmax(PCE_arr)
            #     rotation = thetas[start_idx + idx]
            #     matrix_off = matrix[start_idx + idx]
            #     modifiedcheck = True
        
        PCE_MAX_ARR = [np.max(x) for x in PCE_arr]
        PCE_MAX_ARR = [0 if math.isnan(x) else x for x in PCE_MAX_ARR]
        idx_max = np.where(PCE_MAX_ARR == np.max(PCE_MAX_ARR))
        i = idx_max[0][0]
        idx = np.where(PCE_arr[i] == np.max(PCE_arr[i]))
        if np.max(PCE_arr[i]) > bestpce:
            bestpce = np.max(PCE_arr[i])
            rotation = thetas[i * batch_size + idx[0][0]]
            matrix_off = matrix[i * batch_size + idx[0][0]]
            modifiedcheck = True
        
        del PCE_arr, batchW
    
    else:
        XC = crosscorr_Fingeprint_GPU(batchW[0:len(matrix)], TA, norm2, size_Fingeprint)
        PCE_arr0 = parallel_PCE(XC.numpy(), len(batchW[0:len(matrix)]), ranges[0:len(matrix)])
        del XC
        idx = np.where(PCE_arr0 == np.max(PCE_arr0))
        if np.max(PCE_arr0) > bestpce:
            bestpce = np.max(PCE_arr0)
            rotation = thetas[idx[0][0]]
            matrix_off = matrix[idx[0][0]]
            modifiedcheck = True
        del batchW, PCE_arr0
        
    # scaling estimation
    matrix, ranges, arr_scale = scale_matrix_estimator(
        hom, noise.shape, centerres, step, rotation, size_Fingeprint
    )
    if len(matrix) != len(list_Wrs):
        del list_Wrs
        list_Wrs, batchW = utils.warp(noise, matrix, size_Fingeprint)
    else:
        batchW = tfa.image.transform(list_Wrs, matrix, 'BILINEAR', [size_Fingeprint[1], size_Fingeprint[2]])
    XC = (crosscorr_Fingeprint_GPU(batchW, TA, norm2, size_Fingeprint))
    PCE_arr0 = parallel_PCE(XC.numpy(), len(batchW), ranges)
    del XC, batchW
    pceres = np.max(PCE_arr0)
    idx = np.where(PCE_arr0 == pceres)
    del PCE_arr0
    
    if pceres > bestpce:
        bestpce = pceres
        scaling = arr_scale[idx[0][0]] - 1
        modifiedcheck = True
        matrix_off = matrix[idx[0][0]]
    if step < 3 and modifiedcheck:
        matrix_off, bestpce, rotation, scaling = calibration_GPU(
            hom, noise, rotation, scaling, step + 1, TA, norm2, size_Fingeprint, matrix_off, bestpce
        )
    if step == 0:
        W_T = tf.expand_dims(
            tfa.image.transform(list_Wrs[0], matrix_off, 'BILINEAR', [size_Fingeprint[1], size_Fingeprint[2]]), axis=0
        )
        XC = (crosscorr_Fingeprint_GPU(W_T, TA, norm2, size_Fingeprint))
        ranges = [[(size_Fingeprint[1] - np.round(noise.shape[0] / (scaling + 1))).astype(int),
                   (size_Fingeprint[2] - np.round(noise.shape[1] / (scaling + 1))).astype(int)]]
        bestpce = parallel_PCE(XC.numpy(), len(W_T), ranges)
    del list_Wrs
    return matrix_off, bestpce, rotation, scaling


def rotation_matrix_estimator(hom, noise_shape, centerrot, centerres, step):
    idx_hom = 0
    rotation_arr = sorted(
        [i for i in np.arange(
            -((5 if step == 0 else 0.5) * (10 ** -step)),
            ((5 if step == 0 else 0.5) * (10 ** -step)),
            0.1 * (10 ** -step)
        )], key=abs
    )
    if not np.any(hom):
        hom = np.zeros([len(rotation_arr), 3, 3])
    else:
        hom = np.repeat(hom, repeats=len(rotation_arr), axis=0)
    matrix = np.zeros([len(rotation_arr), 3, 3])
    arr_rotation = []
    for i in rotation_arr:
        matrix[idx_hom] = hom[idx_hom] + np.r_[
            cv2.getRotationMatrix2D((noise_shape[0] / 2, noise_shape[1] / 2), 2 * (centerrot - i), 1.0), [[0, 0, 1]]]
        arr_rotation.append(centerrot - i)
        matrix[idx_hom] = matrix[idx_hom] / (1 + centerres)
        matrix[idx_hom, 2, 2] = matrix[idx_hom, 2, 2] * (1 + centerres)
        idx_hom += 1
    mat_reshape = matrix.reshape([len(rotation_arr), 9])
    return mat_reshape[:, 0:8], arr_rotation


def scale_matrix_estimator(hom, noise_shape, centerres, step, rotation, K_shape):
    scale_arr = sorted(np.arange(-(0.05 * (10 ** -step)), (0.05 * (10 ** -step)), 0.01 * (10 ** -step)), key=abs)
    idx_hom = 0
    if not np.any(hom):
        hom = np.zeros([len(scale_arr), 3, 3])
    else:
        hom = np.repeat(hom, repeats=len(scale_arr), axis=0)
    matrix = np.zeros([len(scale_arr), 3, 3])
    ranges = []
    arr_scale = []
    for i in scale_arr:
        scale = (1 + centerres + i)
        arr_scale.append(scale)
        ranges.append(
            [(K_shape[1] - np.round(noise_shape[0] / scale)).astype(int),
             (K_shape[2] - np.round(noise_shape[1] / scale)).astype(int)]
        )
        matrix[idx_hom] = hom[idx_hom] + np.r_[
            cv2.getRotationMatrix2D((noise_shape[0] / 2, noise_shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
        matrix[idx_hom] = matrix[idx_hom] / scale
        matrix[idx_hom, 2, 2] = matrix[idx_hom, 2, 2] * scale
        idx_hom += 1
    mat_reshape = matrix.reshape([len(scale_arr), 9])
    return mat_reshape[:, 0:8], ranges, arr_scale
