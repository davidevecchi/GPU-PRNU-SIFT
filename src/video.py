import os

import cv2
import exiftool
import numpy as np

from src.utils import Method, Mode
from src.frame_selector import FrameSelectorRaft, FrameSelectorSift


class Video:
    def __init__(self, path, method, mode):
        self.path = path
        self.method = method
        self.mode = mode
        
        self.cap = None
        self.flip = False
        self.rotation = 0
        
        self.I_index = None
        self.pce_index = None  # fixme rename
        self.pce_index_name = self.method.name + '_index'
        
        self.npz_dict = dict()
        self.npz_file = os.path.join('video_data', os.path.splitext(os.path.basename(self.path))[0] + '.npz')
        self.npz_exists = os.path.exists(self.npz_file)
        
        if self.npz_exists:
            self.load_npz()
        else:
            self.get_I_frames_index()
            self.get_rotation()
        if 1 or self.pce_index is None:
            self.set_pce_index()
        
        print('Rotation: %d' % self.rotation)
        self.save_npz()
    
    def set_pce_index(self):
        if self.method == Method.NEW:
            self.pce_index = self.I_index
        elif self.method == Method.NO_INV:
            self.pce_index = []
            end = len(self.I_index) - 1  # fixme 8
            for i in range(len(self.I_index[:end])):
                idx0 = self.I_index[i]
                idx1 = self.I_index[i + 1]
                self.pce_index.append(idx0)
                self.pce_index.append(idx0 + 1)
                self.pce_index.append((idx0 + idx1) // 2)
                self.pce_index.append(idx1 - 1)
            self.pce_index.append(self.I_index[end])
        print(f'pce_index: {self.pce_index}')
        print(f'len(pce_index): {len(self.pce_index)}')
    
    def load_npz(self):
        print(f'Loading {self.npz_file}:', end=' ')
        with np.load(self.npz_file) as npz:
            self.I_index = npz['I_index']
            if self.pce_index_name in npz:
                self.pce_index = npz[self.pce_index_name]
            self.rotation = npz['rotation'][0]
            self.flip = self.rotation == 180
            for key in npz:
                print(key, end=' ')
                self.npz_dict[key] = npz[key]
            print()
    
    def save_npz(self):
        self.npz_dict['I_index'] = self.I_index
        self.npz_dict['rotation'] = [self.rotation]
        if self.pce_index is not None:
            self.npz_dict[self.pce_index_name] = self.pce_index
        np.savez(self.npz_file, **self.npz_dict)
    
    def get_rotation(self):
        with exiftool.ExifTool() as et:
            self.rotation = et.get_metadata(self.path)['Composite:Rotation']
        self.flip = self.rotation == 180
    
    def get_I_frames_index(self):
        path_to_file = 'frames.txt'
        os.system('ffprobe %s -show_frames 2> /dev/null | grep -E pict_type > %s' % (self.path, path_to_file))
        f = open(path_to_file, 'r')
        lines = f.readlines()
        self.I_index = []
        for i, line in enumerate(lines):
            if line[-2] == 'I':
                self.I_index.append(i)
        os.system('rm -r ' + path_to_file)
    
    def select_anchors(self):
        if self.method == Method.ICIP:
            frame_selector = FrameSelectorSift(self, self.mode == Mode.SKIP_GOP0)
        else:
            frame_selector = FrameSelectorRaft(self)
        first_anchor, second_anchor = frame_selector.get_less_stabilized_anchors()
        self.reset_cap()
        return first_anchor, second_anchor
    
    def capture_frame(self, frame_idx, revert_rotation=True):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if self.flip and revert_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return ret, frame
    
    def reset_cap(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.path)
