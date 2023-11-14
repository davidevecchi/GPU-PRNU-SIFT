import glob
import os
import random
import shutil
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.io import loadmat

from src import utils, correlation
from src.device_data import DeviceData
from src.utils import Method, Mode
from src.video import Video

METHOD = Method.NEW
MODE = Mode.ALL


class FingerprintAnalyzer:
    def __init__(self, loader, fingerprint_path: str):
        self.loader = loader
        self.fingerprint_path = fingerprint_path
        self.out_file_basename = None
        self.video_paths = None
        self.video = None
        
        self.device = None
        self.device0 = None
        
        self.TA_tf = None
        self.norm2 = None
        self.tilted_array2 = None
        self.size_fing = None
        
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    def run_analysis(self):
        self.load_device()
        self.load_camera_fingerprint()
        
        prefix = (self.device.id + '_') if self.loader.hypothesis == 0 else ''
        for i, video_path in enumerate(self.video_paths):
            self.out_file_basename = os.path.join(self.loader.output_path, prefix + os.path.basename(os.path.splitext(video_path)[0]))
            print(f'\n{i + 1}/{len(self.video_paths)}\t' + self.out_file_basename)
            
            if not os.path.exists(self.out_file_basename + '.npz'):  # fixme
                self.video = Video(video_path, METHOD, MODE)
                if self.video.rotation in [90, 270]:
                    print('vertical video, skipping')
                    shutil.move(self.video.path, self.video.path.replace('dataset', 'skip'))
                else:
                    if self.video.pce_index is None:
                        index = self.select_index()
                        self.video.pce_index = index
                        self.video.save_npz()
                    pce_array, time_array = self.process_video()
                    self.save(pce_array, time_array)  # fixme
                self.video.cap.release()
    
    def load_device(self):
        print()
        self.device = DeviceData(self.fingerprint_path)
        self.device0 = None
        choice = None
        if self.loader.hypothesis == 0:
            flag_choice = True
            while flag_choice:
                # FIXME!!!!!!!!!!!!!!!!!
                rnd_idx = random.randint(0, len(self.loader.fingerprint_paths) - 1)
                choice = self.loader.fingerprint_paths[rnd_idx]
                # choice = 'PRNU_fingerprints/Fingerprint_%s.mat' % ('D25' if 'D06' in self.fingerprint_path else 'D06')
                # FIXME!!!!!!!!!!!!!!!!!
                
                print(choice, self.fingerprint_path)
                if choice != self.fingerprint_path:
                    flag_choice = False
            self.device0 = DeviceData(self.fingerprint_path, choice)
            print('device 0:', self.device0.id)
        test_set_device = os.path.join(self.loader.test_set, (self.device if self.loader.hypothesis == 1 else self.device0).id + '*')
        self.video_paths = sorted(glob.glob(test_set_device))
        print('device 1:', self.device.id)
        print('-------------')
    
    def load_camera_fingerprint(self):
        K = loadmat(self.fingerprint_path)
        Fingerprint = K['fing']
        Fingerprint = cv2.resize(Fingerprint, (0, 0), fx=(1 / self.device.basescaling), fy=(1 / self.device.basescaling))
        Fingerprint = Fingerprint[self.device.crop_array[0]:self.device.crop_array[1], self.device.crop_array[2]:self.device.crop_array[3]]
        self.size_fing = np.shape(Fingerprint)
        array2 = Fingerprint.astype(np.double)
        array2 = array2 - array2.mean()
        self.tilted_array2 = np.fliplr(array2)
        self.tilted_array2 = np.flipud(self.tilted_array2)
        self.norm2 = np.sum(np.power(array2, 2))
        TA = np.fft.fft2(self.tilted_array2)
        self.TA_tf = tf.expand_dims(tf.convert_to_tensor(TA, dtype=tf.complex64), axis=0)
    
    def select_index(self):
        print('Selecting frame range...')
        start = time.perf_counter()
        first_anchor, second_anchor = self.video.select_anchors()
        print(f'Anchors: {first_anchor}-{second_anchor}')
        index = self.select_index_order(first_anchor, second_anchor)
        print('TIME CONSUMED frame selector:', time.perf_counter() - start)
        return index
    
    def select_index_order(self, first_anchor, second_anchor):
        _, first_anchor_frame = self.video.capture_frame(first_anchor)
        _, second_anchor_frame = self.video.capture_frame(second_anchor)
        noise, W_T1 = utils.get_noise_WT(first_anchor_frame, self.size_fing)
        noise, W_T2 = utils.get_noise_WT(second_anchor_frame, self.size_fing)
        
        XC = correlation.crosscorr_Fingeprint_GPU(
            tf.expand_dims([W_T1, W_T2], axis=3), self.TA_tf, self.norm2, (1,) + np.shape(self.tilted_array2)
        )
        ranges = [[(self.size_fing[0] - noise.shape[0]), (self.size_fing[1] - noise.shape[1])],
                  [(self.size_fing[0] - noise.shape[0]), (self.size_fing[1] - noise.shape[1])]]
        pce_anchors = correlation.parallel_PCE(XC.numpy(), len(XC), ranges)
        
        index = list(range(first_anchor, second_anchor + 1))
        if pce_anchors[1] > pce_anchors[0]:
            index = list(reversed(index))
        
        return index
    
    def process_video(self):
        TARGET_PCE = 50000000000000  # FIXME
        MAX_HITS = 3000000000000  # FIXME
        MIN_PCE = 30000000000000  # FIXME
        MAX_ADDED = 12
        
        hit_frames = 0
        i_frame_idx = 1 if MODE == Mode.SKIP_GOP0 else 0
        p_frame_idx = 1 if MODE == Mode.SKIP_I0 else 0
        
        index = self.video.pce_index
        time_array = []
        pce_array = []
        oframe = None
        
        def next_idx(i_idx, p_idx, _index):
            if i_idx + 1 < len(_index) and _index[i_idx] + p_idx >= _index[i_idx + 1]:
                i_idx += 1
                p_idx = 0
            else:
                p_idx += 1
            return i_idx, p_idx
        
        self.video.reset_cap()
        
        start_run = time.perf_counter()
        go = True
        while go:
            if METHOD == Method.NEW:
                frame_idx = index[i_frame_idx] + p_frame_idx
                IP_type = 'I' if p_frame_idx == 0 else 'P'
                print(f'{IP_type}-frame:', frame_idx, f'\taccepted {len(pce_array)}/{MAX_ADDED}', end='\t')
            else:
                frame_idx = index[p_frame_idx]
                print('frame:', frame_idx, end='\t')
            
            ret, frame = self.video.capture_frame(frame_idx)
            
            if not ret:
                print('Video ended')
                break
            
            pce, oframe = self.get_best_pce(frame, oframe)
            print('[%.2f]' % pce)
            
            if METHOD == Method.NEW:
                if pce >= MIN_PCE:
                    i_frame_idx, p_frame_idx = next_idx(i_frame_idx, p_frame_idx, index)
                    pce_array.append(pce)
                else:
                    if p_frame_idx == 0:
                        pce_array.append(pce)
                    i_frame_idx += 1
                    p_frame_idx = 0
                    oframe = None  # reset oframe if frames are skipped
                go = len(pce_array) < MAX_ADDED and hit_frames < MAX_HITS and i_frame_idx < len(index)
            else:
                pce_array.append(pce)
                p_frame_idx += 1
                go = p_frame_idx < len(index)
            
            if go and pce >= TARGET_PCE and METHOD == Method.NEW:
                hit_frames += 1
                go = go and hit_frames < MAX_HITS
            
            time_array.append(time.perf_counter() - start_run)
        
        print('MEAN PCE:', '%.2f' % np.mean(pce_array))
        
        return pce_array, time_array
    
    def get_best_pce(self, frame, oframe):
        noise, W_T = utils.get_noise_WT(frame, self.size_fing)
        pce_res = self.get_pce(W_T, noise)
        print('PCE resizing: %.2f' % pce_res, end='\t')
        
        if METHOD == Method.NO_INV:
            return pce_res, None
        
        printed = False
        try:
            assert oframe is not None
            pce_corr, H = self.pce_correction(frame, oframe, noise)
            print('PCE correction: %.2f' % pce_corr, end='\t  ')
            printed = True
            assert pce_corr > pce_res
            pce, oframe = self.pce_calibration(noise, pce_corr, frame, H, self.loader.hypothesis)
        except (ValueError, AssertionError):
            if not printed:
                print(' ' * 21, end='\t  ')
            pce, oframe = self.pce_calibration(noise, pce_res, frame)
        
        return pce, oframe
    
    def pce_correction(self, frame, oframe, noise):
        orb = cv2.SIFT_create()
        queryKeypoints, queryDescriptors = orb.detectAndCompute(frame, None)
        trainKeypoints, trainDescriptors = orb.detectAndCompute(oframe, None)
        
        if queryDescriptors is None or trainDescriptors is None:
            raise ValueError('Query or train descriptors are missing')
        
        matches = self.matcher.match(queryDescriptors, trainDescriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.9)]
        
        if matches:
            matches = utils.get_matches(queryKeypoints, trainKeypoints, matches)
        
        if len(matches) <= 4:
            raise ValueError('Less than 4 matches found')
        
        p1 = np.zeros((len(matches), 2))
        p2 = np.zeros((len(matches), 2))
        for i in range(len(matches)):
            p1[i, :] = queryKeypoints[matches[i].queryIdx].pt
            p2[i, :] = trainKeypoints[matches[i].trainIdx].pt
        
        # FAST
        H, mask = cv2.findHomography(p1, p2, cv2.USAC_DEFAULT)
        if H is None or (self.loader.hypothesis == 0 and len(matches) <= 50):
            raise ValueError('homography_matrix is None or (hypothesis == 0 and len(matches) <= 50)')
        
        H[0, 2] = 0
        H[1, 2] = 0
        noisecorr = cv2.warpPerspective(noise, H, (np.shape(noise)[1], np.shape(noise)[0]))
        W_corr = tfa.image.transform(
            tf.convert_to_tensor(noisecorr, dtype=tf.float32), [1, 0, 0, 0, 1, 0, 0, 0], 'BILINEAR', [self.size_fing[0], self.size_fing[1]]
        )
        pce_corr = self.get_pce(W_corr, noisecorr)
        return pce_corr, H
    
    def get_pce(self, W_T, noise):
        XC = correlation.crosscorr_Fingeprint_GPU(
            tf.expand_dims(tf.expand_dims(W_T, axis=0), axis=3), self.TA_tf, self.norm2, (1,) + np.shape(self.tilted_array2)
        )
        ranges = [[(self.size_fing[0] - noise.shape[0]),
                   (self.size_fing[1] - noise.shape[1])]]
        pce_res = correlation.parallel_PCE(XC.numpy(), len(XC), ranges)
        return pce_res
    
    def pce_calibration(self, noise, pce, resized, H=None, hypothesis=None):  # fixme rename
        H = np.zeros((3, 3)) if H is None else np.linalg.inv(H)
        size_fingerprint = (1,) + np.shape(self.tilted_array2) if hypothesis is None or hypothesis == 1 else np.shape(self.TA_tf)
        H1, pce1, rotation, scaling = correlation.calibration_GPU(
            H, noise, 0, 0, 0, self.TA_tf, self.norm2, size_fingerprint, [1, 0, 0, 0, 1, 0, 0, 0], pce
        )
        oframe = tfa.image.transform(
            resized, H1, 'BILINEAR',
            [np.uint32(np.rint(resized.shape[0] / H1[0])),
             np.uint32(np.rint(resized.shape[1] / H1[0]))]
        ).numpy()
        return max(pce, pce1), oframe
    
    def save(self, pce_array, time_array):
        if not os.path.exists(self.loader.output_path):
            os.mkdir(self.loader.output_path)
        np.savez(self.out_file_basename, pce=pce_array, time=time_array)
        self.video.save_npz()
        # mdir = {'pce': np.asarray(pce_array)}
        # out_name1 = self.out_file_basename + '_PCE.mat'
        # savemat(out_name1, mdir)
        # mdir = {'time': np.asarray(time_array)}
        # out_name2 = self.out_file_basename + '_time.mat'
        # savemat(out_name2, mdir)


"""
 hit_frames = 0
 MIN_PCE = 30
 TARGET_PCE = 100
 MAX_HITS = 4
 MAX_ADDED = 12
 i_frame_idx = 0
 p_frame_idx = 0
 pce_array = []
 index = video.get_I_frames()

 def next_idx(i_idx, p_idx, _index):
     if i_idx + 1 < len(_index) and _index[i_idx] + p_idx >= _index[i_idx + 1]:
         i_idx += 1
         p_idx = 0
     else:
         p_idx += 1
     
     return i_idx, p_idx

 while len(pce_array) < MAX_ADDED and hit_frames < MAX_HITS:
     frame_idx = index[i_frame_idx] + p_frame_idx
     ret, resized = video.capture_frame(frame_idx)
     if not ret: break
     
     PCE = compute_best_pce(...)  # best PCE after resizing, correction and calibration
     
     # if the frame, I or P type, has PCE >= MIN_PCE, add it to the list and move on to the next P-frame
     if PCE >= MIN_PCE:
         pce_array.append(PCE)
         if PCE >= TARGET_PCE: hit_frames += 1
         i_frame_idx, p_frame_idx = next_idx(i_frame_idx, p_frame_idx, index)
     # if the frame is of type I, with PCE < MIN_PCE, add it to the list and skip to the next I-frame
     elif p_frame_idx == 0:
         pce_array.append(PCE)
         i_frame_idx += 1
     # if the frame is of type P, with PCE < MIN_PCE, ignore it and skip to the next I-frame
     else:
         i_frame_idx += 1
         p_frame_idx = 0
     
     if i_frame_idx == len(index): break

 return pce_array
"""
