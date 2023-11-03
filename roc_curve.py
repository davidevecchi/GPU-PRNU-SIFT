import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

TARGET_FPR = 0.05

YMIN = 0.7
XMAX = 1.0
TICKS = 7

cm = plt.get_cmap('tab20')


def get_avg_pce(folder, func, device=''):
    files = sorted(os.listdir(folder))
    mean_pce = []
    for file in files:
        if (file.endswith('.npz') and (device == '' or file.startswith(device))
                and (not skip_flat_still or ('flat' not in file and 'still' not in file))):
            with np.load(os.path.join(folder, file)) as npz:
                pce_mat = npz['pce']
            mean = func(np.array(pce_mat).flatten())
            mean_pce.append(mean)
    return np.array(mean_pce)


P_pce = None


def plt_hist(folder):
    global P_pce
    files = os.listdir(folder)
    I_pce = np.empty(0)
    P_pce = np.empty(0)
    
    count = {key[4:7]: 0 for key in files}
    
    idxs = []
    pce_y = []
    
    content = {'flat': [], 'indoor': [], 'outdoor': []}
    movement = {'still': [], 'move': [], 'panrot': []}
    
    content = 'flat'
    
    for file in files:
        if file.endswith('.npz'):
            with np.load(os.path.join(folder, file)) as npz:
                pce_mat = npz['pce'].flatten()
            
            # if file[4:7] == 'D15':
            if content in file:
                I_pce = np.concatenate((I_pce, np.array([pce_mat[0], pce_mat[-1]])))
                P_pce = np.concatenate((P_pce, pce_mat[1:-1]))
                
                # I_pce = np.concatenate((I_pce, pce_mat[0::2]))
                # P_pce = np.concatenate((P_pce, pce_mat[1::2]))
                pass
            
            # P_pce = np.concatenate((P_pce, pce_mat))
    
    #         for i, pce in enumerate(pce_mat):
    #             idxs.append(i)
    #             pce_y.append(pce)
    #         count[file[4:7]] += (pce_mat < 20).sum()
    # print(count)
    
    # plt.scatter(pce_y, idxs, alpha=.1)
    # print(np.percentile(P_pce, 95))
    # print(np.percentile(P_pce, 99.9))
    plt.hist(P_pce, bins=128)
    plt.hist(I_pce, bins=128)
    plt.title(content)
    plt.xlim([0, 60])
    plt.show()
    input()


skip_flat_still = False


def plot_roc(axs, folder_h0, folder_h1, label, func=np.mean, device='', bar=False, **kwargs):
    pce_h0 = get_avg_pce(folder_h0, func)
    pce_h1 = get_avg_pce(folder_h1, func, device)
    
    labels = [0] * len(pce_h0) + [1] * len(pce_h1)
    all_pce = np.concatenate([pce_h0, pce_h1])
    fpr, tpr, thresholds = roc_curve(labels, all_pce, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    
    tau_idx = (np.abs(fpr - TARGET_FPR)).argmin()
    tau = thresholds[tau_idx]
    
    # if 'lw' not in kwargs:
    #     kwargs['lw'] = 1
    axs[0].plot(
        fpr, tpr, **kwargs,
        label='[AUC=%0.3f; TPR=%0.3f @ fpr=%0.4f, tau=%0.1f] %s' % (roc_auc, tpr[tau_idx], fpr[tau_idx], tau, label + ' ' + device)
    )
    
    width = 0.3  # the width of the bars
    gap = 0.05
    multiplier = [-1, 1]
    new = plot_roc.x > int(plot_roc.x)
    
    def formatt(y):
        return ('%.3f' % y)[1:] if y < .9999 else '1.00'
    
    if not bar:
        return
    
    for i, y, l in zip(multiplier, [roc_auc, tpr[tau_idx]], ['auc', 'tpr']):
        offset = (width + gap) * i * 0.5
        if new:
            kwargs['color'] = cm(1)
            kwargs['zorder'] = -1
        rects = axs[1].bar(int(plot_roc.x) + offset, y, width, label=l, **kwargs)
        axs[1].bar_label(rects, labels=[formatt(y)], padding=3 if new else -15)
    plot_roc.x += 0.5


plot_roc.x = 0


def max_mean_mean(arr):
    return np.mean(arr[arr > np.mean(arr)])


def max_med_mean(arr):
    return np.median(arr[arr > np.mean(arr)])


def max_mean_med(arr):
    return np.mean(arr[arr > np.median(arr)])


def square(arr):
    return np.mean(arr ** 4)


def plt_setup():
    # fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(16, 16), dpi=300, sharey='row')
    
    fig = plt.figure(constrained_layout=True, figsize=(16, 16), dpi=300)
    subfigs = fig.subfigures(nrows=2, ncols=1)
    axs = np.empty((2, 2), dtype=object)
    titles = ['Receiver Operating Characteristic', 'AUC / TPR comparison']
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(titles[i])
        ax = subfig.subplots(nrows=1, ncols=2, sharey='row', )
        axs[i, 0], axs[i, 1] = ax[0], ax[1]
    
    axc = [None, None]
    for i in range(2):
        axs[0, i].plot([TARGET_FPR, TARGET_FPR], [0.0, 1.0], lw=0.5, linestyle='--', color='gray')
        axs[0, i].plot([0, 1], [1.0, 1.0], lw=0.5, linestyle='--', color='gray')
        axs[0, i].set_xlim([0.0, XMAX])
        axs[0, i].set_ylim([YMIN, 1.02])
        axs[0, i].set_yticks(np.linspace(YMIN, 1, TICKS))
        axs[0, i].set_xlabel('False Positive Rate')
        
        axs[1, i].grid(visible=True, axis='y', lw=0.5, linestyle='--')
        axs[1, i].set_ylim([YMIN, 1.02])
        axs[1, i].set_yticks(np.linspace(YMIN, 1, TICKS))
        axs[1, i].set_xlabel('Metrics, Devices')
        
        # fixme: add fake label '1.0' at the SE corner of the second row, to match the width of the first row
        axc[i] = axs[1, i].twiny()
        axc[i].set_xticks([1, 1.1])
        axc[i].set_xlim([0, 1])
        axc[i].tick_params(bottom=False, top=False, labelbottom=True, labeltop=False, labelcolor='w')
    
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[1, 0].set_ylabel('AUC / TPR @ fpr=0.05')
    
    return axs


def percentile_weight(arr):
    e = 4
    return np.mean(arr ** e) ** (1 / e)
    
    weights = []
    arr2 = np.concatenate((arr, P_pce))
    for x in arr:
        p = np.sum(P_pce < x) / len(P_pce)
        weights.append(p)
    mean = np.mean(arr * weights * 2)
    print(*['%.3f' % x for x in arr], sep='\t')
    print(*['%.3f' % x for x in weights], sep='\t')
    print(mean)
    print()
    return mean


def main():
    global skip_flat_still
    folder_h0_old = 'results/results_h0'
    folder_h0_new = 'results/results_h0_new'
    
    folder_h1_sift_____ = 'results/results_h1_sift'
    folder_h1_sift_I0__ = 'results/results_h1_sift_I0'
    folder_h1_sift_GOP0 = 'results/results_h1_sift_GOP0'
    folder_h1_new______ = 'results/results_h1_new'
    folder_h1_new_I0___ = 'results/results_h1_new_I0'
    folder_h1_new_GOP0_ = 'results/results_h1_new_GOP0'
    folder_h1_raft_____ = 'results/results_h1_raft'
    
    plt_hist(folder_h0_old)
    
    # plot_roc(folder_h0_old, folder_h1_sift_____, 'SIFT skip', np.mean, True)
    # plot_roc(folder_h0_new, folder_h1_new______, 'new skip', np.mean, True)
    devices = ['D06', 'D14', 'D15', 'D19', 'D25']
    devices = ['D06', 'D25']
    
    axss = plt_setup()
    
    # print(*get_avg_pce(folder_h1_new______, np.mean, 'D25').astype(int))
    # print(*get_avg_pce(folder_h1_new_GOP0_, np.mean, 'D25').astype(int))
    # print(*get_avg_pce(folder_h1_new_I0___, np.mean, 'D25').astype(int))
    
    for i in range(2):
        axs = axss[:, i]
        plot_roc(axs, folder_h0_old, folder_h1_sift_____, 'SIFT        ', np.mean, color=cm(3), lw=2)
        # plot_roc(axs, folder_h0_old, folder_h1_sift_I0__, 'SIFT --I_0  ', np.mean, color=cm(1), lw=2)
        # plot_roc(axs, folder_h0_old, folder_h1_sift_GOP0, 'SIFT --GOP_0', np.mean, color=cm(5), lw=2)
        # plot_roc(axs, folder_h0_old, folder_h1_raft_____, 'RAFT        ', np.mean, color=cm(6), lw=2)
        plot_roc(axs, folder_h0_new, folder_h1_new______, 'NEW         ', np.mean, color=cm(2), lw=2)
        plot_roc(axs, folder_h0_new, folder_h1_new______, 'NEW perc    ', percentile_weight, color=cm(1), lw=2)
        plot_roc(axs, folder_h0_new, folder_h1_new______, 'NEW max     ', np.max, color=cm(0), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new_I0___, 'NEW --I_0   ', np.mean, color=cm(0), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new_GOP0_, 'NEW --GOP_0 ', np.mean, color=cm(4), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new______, 'SQUARE      ', np.mean, color=cm(8), lw=2)
        # for j, devide in enumerate(devices):
        #     plot_roc(axs, folder_h0_new, folder_h1_new______, 'SIFT', np.mean, device=devide, lw=2, color=cm((j + 4) % 20), linestyle='--')
        #     plot_roc(axs, folder_h0_new, folder_h1_new______, 'new', percentile_weight, device=devide, lw=2, color=cm((j + 4) % 20))
        
        axss[0, i].legend(loc='lower right')
        bar_labels = 'AUC, TPR\n%s'
        ticks = [bar_labels % 'ALL']
        for device in devices:
            ticks += [bar_labels % device]
        axss[1, i].set_xticks(range(len(devices) + 1), ticks)
        
        plot_roc.x = 0
        skip_flat_still = True
    
    plt.show()


if __name__ == '__main__':
    main()
