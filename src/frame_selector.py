import math
import os.path
import statistics
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.transforms import functional

from src import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class FrameSelector(ABC):
    
    def __init__(self, video, skip_gop0=False):
        self.video = video
        self.skip_gop0 = skip_gop0
    
    @abstractmethod
    def get_anchor(self):
        pass
    
    def get_less_stabilized_anchors(self):
        anchor = self.get_anchor()
        return self.video.I_index[anchor], self.video.I_index[anchor + 1]


class FrameSelectorSift(FrameSelector):
    INF = 999999
    
    def __init__(self, video, skip_gop0=False):
        super().__init__(video, skip_gop0)
        self.momentum_array = []
    
    def get_anchor(self):
        self.compute_momentums()
        
        min_momentum_idx = np.argmin(self.momentum_array)
        print(f'min_momentum: {self.momentum_array[min_momentum_idx]}   min_momentum_idx: {min_momentum_idx}')
        
        if not self.skip_gop0 and self.momentum_array[min_momentum_idx] > 10:
            min_momentum_idx = 0
        if self.skip_gop0 and self.momentum_array[min_momentum_idx] == self.INF:
            min_momentum_idx = 1
        
        return min_momentum_idx
    
    def compute_momentums(self):
        ret, oframe = self.video.capture_frame(self.video.I_index[0], revert_rotation=False)
        trainKeypoints, trainDescriptors = None, None
        for idx in self.video.I_index[1:]:
            ret, frame = self.video.capture_frame(idx, revert_rotation=False)
            if not ret:
                break
            
            orb = cv2.SIFT_create()
            queryKeypoints, queryDescriptors = orb.detectAndCompute(frame, None)
            if oframe is not None:
                trainKeypoints, trainDescriptors = orb.detectAndCompute(oframe, None)
            
            if not (queryDescriptors is None or trainDescriptors is None):
                matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                matches = matcher.match(queryDescriptors, trainDescriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                matches = matches[:int(len(matches) * 0.9)]
                if len(matches):
                    matches = utils.get_matches(queryKeypoints, trainKeypoints, matches)
                    avg_mov = statistics.mean(
                        [math.sqrt(
                            (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 +
                            (queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[1]) ** 2
                        ) for match in matches]
                    )
                    self.momentum_array.append(avg_mov)
                else:
                    self.momentum_array.append(self.INF)
                trainKeypoints = queryKeypoints
                trainDescriptors = queryDescriptors
                oframe = None
            else:
                self.momentum_array.append(self.INF)
                oframe = frame
        
        self.video.cap.release()


class FrameSelectorRaft(FrameSelector):
    torchvision.set_video_backend('video_reader')
    SIZE = [520, 960]
    
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    
    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    
    def __init__(self, video, skip_gop0=False):
        super().__init__(video, skip_gop0)
        self.flow_move_avg = []
        self.diff_move_avg = []
        self.flow_med = []
        self.diff_med = []
        self.prod_med = []
        
        self.save_folder = 'raft_results'
        self.npz_file = os.path.join(self.save_folder, os.path.splitext(os.path.basename(self.video.path))[0] + '_raft.npz')
    
    @staticmethod
    def preprocess(img1_batch, img2_batch):
        img1_batch = functional.resize(img1_batch, size=FrameSelectorRaft.SIZE, antialias=False)
        img2_batch = functional.resize(img2_batch, size=FrameSelectorRaft.SIZE, antialias=False)
        return FrameSelectorRaft.transforms(img1_batch, img2_batch)
    
    def get_anchor(self):
        if os.path.exists(self.npz_file):
            self.load_npz()
        else:
            self.compute_move_avg()
            np.savez(self.npz_file, flow=self.flow_move_avg, diff=self.diff_move_avg)
        return self.compute_momentum()
    
    def load_npz(self):
        npz = np.load(self.npz_file)
        self.flow_move_avg = npz['flow']
        self.diff_move_avg = npz['diff']
    
    def compute_move_avg(self):
        torch.cuda.empty_cache()
        print('Loading frames from', self.video.path, end=' ')
        frames, _, _ = read_video(str(self.video.path), output_format='TCHW', pts_unit='sec')
        print('ok')
        
        prev_flow = None
        batch_size = 2
        
        for i in range(0, len(frames), batch_size):
            anchor0 = i
            anchor1 = min(i + batch_size, len(frames) - 1)
            
            if anchor1 == anchor0:
                break
            
            print(f'raft: {anchor0}/{len(frames)}', end='\r')
            
            img1_batch = torch.stack(tuple(frames[anchor0:anchor1]))
            img2_batch = torch.stack(tuple(frames[anchor0 + 1:anchor1 + 1]))
            img1_batch, img2_batch = FrameSelectorRaft.preprocess(img1_batch, img2_batch)
            
            list_of_flows = FrameSelectorRaft.model(img1_batch.to(device), img2_batch.to(device))
            predicted_flows = list_of_flows[-1]
            
            for j in range(len(predicted_flows)):
                flow = predicted_flows[j]
                flow_move = torch.sqrt(flow[0] ** 2 + flow[1] ** 2)
                self.flow_move_avg.append(torch.median(flow_move).item())
                
                if prev_flow is not None:
                    diff = flow - prev_flow
                    diff_move = torch.sqrt(diff[0] ** 2 + diff[1] ** 2)
                    self.diff_move_avg.append(torch.median(diff_move).item())
                
                prev_flow = predicted_flows[j]
        
        torch.cuda.empty_cache()
        print(f'raft: {len(frames)}/{len(frames)}')
    
    def compute_momentum(self):  # fixme rename
        for i in range(len(self.video.I_index) - 1):
            anchor0 = int(self.video.I_index[i])
            anchor1 = int(self.video.I_index[i + 1])
            
            self.flow_med.append(np.mean(self.flow_move_avg[anchor0:anchor1 - 1]))
            self.diff_med.append(np.median(self.diff_move_avg[anchor0:anchor1 - 2]))
            self.prod_med.append(self.flow_med[-1] ** 2 * self.diff_med[-1])
            
            """
            I-type :    I           I       I           I
            frame  :   (0)  1   2  (3)  4  (5)  6   7  (8)
            raft   :     [0   1   2] [3   4] [5   6   7]
            diff   :       [0   1]  2  [3]  4  [5   6]
            """
        
        print(np.min(self.prod_med), np.argmin(self.prod_med), self.prod_med)
        
        return np.argmin(self.diff_med)
    
    def plot(self):
        self.load_npz()
        self.compute_momentum()
        
        flow_med_pad = []
        diff_med_pad = []
        prod_med_pad = []
        
        for i in range(len(self.video.I_index) - 1):
            anchor0 = int(self.video.I_index[i])
            anchor1 = int(self.video.I_index[i + 1])
            
            flow_med_pad += [self.flow_med[i]] * (anchor1 - anchor0)
            diff_med_pad += [self.diff_med[i]] * (anchor1 - anchor0)
            prod_med_pad += [self.prod_med[i]] * (anchor1 - anchor0)
        
        fig, ax = plt.subplots(figsize=(80, 12))
        ax.plot(range(len(self.flow_move_avg)), self.flow_move_avg, label='flow')
        ax.plot(range(len(self.diff_move_avg)), self.diff_move_avg, label='diff')
        ax.plot(range(len(prod_med_pad)), prod_med_pad, label='prod med', linewidth=3)
        ax.plot(range(len(diff_med_pad)), diff_med_pad, label='diff med')
        ax.plot(range(len(flow_med_pad)), flow_med_pad, label='flow med')
        ax.set_xticks(self.video.I_index)
        ax.grid(True, which='major', axis='x', linestyle='--')
        plt.tight_layout()
        ax.margins(x=0, y=0)
        ax.set_ylim(0, max(np.max(self.flow_move_avg), np.max(self.diff_move_avg)) + 1)
        ax.set_xlim(0, len(self.flow_move_avg) + 1)
        ax.legend()
        plt.show()


"""
def main():
    video_path = 'D06_V_outdoor_move_0001'
    selector = FrameSelector(video_path)
    selector.npz_file = '../' + selector.npz_file
    selector.plot()


if __name__ == '__main__':
    main()
"""
