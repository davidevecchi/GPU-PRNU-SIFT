import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from matplotlib.ticker import MaxNLocator
from scipy.special import gamma
from sklearn.metrics import auc, roc_curve

from scipy.optimize import minimize
from src.utils import Method

TARGET_FPR = 0.1

XMAX = 1.0
YMIN = 0.67
TICKS = 12

stats_txt = 'plot/stats.txt'
cm = plt.get_cmap('tab20')


def get_avg_pce(folder, func, tau=0, devices=None):
    files = sorted(os.listdir(folder))
    mean_pce = []
    for file in files:
        if (file.endswith('.npz') and (devices is None or file[:3] in devices)
                and (not skip_flat_still or ('flat' not in file and 'still' not in file))):
            with np.load(os.path.join(folder, file)) as npz:
                pce_mat = npz['pce'].flatten()
            thr_pce = pce_mat[pce_mat > tau]
            mean = func(thr_pce)
            mean_pce.append(mean)
    return np.array(mean_pce)


skip_flat_still = False


def plot_roc(axs, folder_h0, folder_h1, label, func=np.mean, devices=None, frame_type=None, bar=False, tau=0, **kwargs):
    pce_h0 = get_avg_pce(folder_h0, func, tau, devices)
    pce_h1 = get_avg_pce(folder_h1, func, tau, devices)
    
    # fixme
    step = 4
    if frame_type == 'I':
        pce_h0 = pce_h0[0::step]
        pce_h1 = pce_h1[0::step]
    elif frame_type == 'P':
        pce_h0 = [pce for i, pce in enumerate(pce_h0) if i % step != 0]
        pce_h1 = [pce for i, pce in enumerate(pce_h1) if i % step != 0]
    
    labels = [0] * len(pce_h0) + [1] * len(pce_h1)
    all_pce = np.concatenate([pce_h0, pce_h1])
    fpr, tpr, thresholds = roc_curve(labels, all_pce, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    
    tau_idx = (np.abs(fpr - TARGET_FPR)).argmin()
    tau = thresholds[tau_idx]
    
    if devices is None:
        devices = ''
    axs[0].plot(
            fpr, tpr, **kwargs,
            label='[AUC=%0.3f; TPR=%0.3f @ fpr=%0.4f, tau=%0.1f] %s' % (roc_auc, tpr[tau_idx], fpr[tau_idx], tau, label + ' ' + str(devices))
    )
    
    width = 0.3
    gap = 0.05
    multiplier = [-1, 1]
    new = plot_roc.x > int(plot_roc.x)
    
    def formatt(val):
        return ('%.3f' % val)[1:] if val < .9999 else '1.00'
    
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


def power(arr):
    e = 4
    return np.mean(arr ** e) ** (1 / e)


def percentile_weights(arr):
    weights = [np.sum(H0_pce_new[H0_pce_new < pce]) / len(H0_pce_new) for pce in arr]
    return np.average(arr, weights=weights)


H0_pce_new = np.empty(0)  # FIXME !!!!!!!


def plt_hist(title, method, devices=None):
    global H0_pce_new
    fig, ax = plt.subplots(4, 1, figsize=(16, 9), tight_layout=True, dpi=300, sharex='col')
    xlim = 200
    bins = np.arange(0, xlim + .5, .5)
    base_title = title
    title += '_ALL' if devices is None else '_' + '-'.join(devices)
    
    for i in [0, 1]:
        folder = f'results/results_h{i}_{base_title}'
        files = os.listdir(folder)
        A_pce_all = np.empty(0)
        I_pce_all = np.empty(0)
        A_pce_avg = np.empty(0)
        
        for file in files:
            if devices is None or file[:3] in devices:
                with np.load(os.path.join(folder, file)) as npz:
                    A_pce_vid = npz['pce'].flatten()
                
                if method == Method.ICIP:
                    I_pce_vid = np.array([A_pce_vid[0], A_pce_vid[-1]])
                elif method == Method.NO_INV:
                    I_pce_vid = A_pce_vid[0::4]
                else:
                    I_pce_vid = np.empty(0)
                
                # A_pce_vid = A_pce_vid[A_pce_vid > 19]
                
                A_pce_all = np.concatenate((A_pce_all, A_pce_vid))
                I_pce_all = np.concatenate((I_pce_all, I_pce_vid))
                A_pce_avg = np.append(A_pce_avg, np.mean(A_pce_vid))
        
        # FIXME !!!!
        if base_title == 'new' and i == 0:
            H0_pce_new = A_pce_all.copy()
            
        med_A, med_I, med_avg, ip0, ip1 = print_stats(A_pce_all, I_pce_all, A_pce_avg, f'h{i}_{title}')
        
        ax[i + 0].hist(A_pce_all, bins=bins, label='all frames PCEs')
        ax[i + 0].hist(I_pce_all, bins=bins, label='I-frames PCEs')
        ax[i + 2].hist(A_pce_avg, bins=bins, color='tab:cyan', label='videos avg PCEs')
        ax[i + 0].axvline(x=med_A, label=f'median all frames = {med_A:.1f}', c='tab:green', lw=1)
        ax[i + 0].axvline(x=med_I, label=f'median I-frames = {med_I:.1f}', c='tab:olive', lw=1)
        ax[i + 2].axvline(x=med_avg, label=f'median videos avg = {med_avg:.1f}', c='tab:pink', lw=1)
        ax[i + 0].set_xlim([0, xlim])
        ax[i + 2].set_xlim([0, xlim])
        ax[i + 2].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i + 2].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i + 0].legend(loc='upper right')
        ax[i + 2].legend(loc='upper right')
        
        ax1 = ax[i + 0].twinx()
        xcum = np.sort(A_pce_all)
        ycum = np.array(range(len(A_pce_all)))
        ax1.plot(xcum, ycum, label='all frames PCEs CDF', color='tab:red')
        ax1.set_ylim([0, ycum[-1]] if len(ycum) > 0 else 0)
        ax1.set_xlim([0, xlim])
        lines0, labels0 = ax[i + 0].get_legend_handles_labels()
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines0 + lines1, labels0 + labels1, loc='upper right')
        
        # ax2 = ax[i + 0].twinx()
        # sample_mean = np.mean(A_pce_all)
        # sample_variance = np.var(A_pce_all, ddof=1)
        # shape, loc, scale = sample_mean ** 2 / sample_variance, 0, sample_variance / sample_mean
        # x = np.linspace(0, xlim, 10 * xlim)
        # y = (x ** (shape - 1)) * np.exp(-x / scale) / (scale ** shape)
        # ax2.plot(x, y, label="Fitted Gamma PDF")

        #
        # sample_mean = np.mean(A_pce_all)
        # sample_variance = np.var(A_pce_all)
        # shape_param = (sample_mean ** 2) / sample_variance
        # scale_param = sample_variance / sample_mean
        # x_values = np.linspace(0, xlim, 10 * xlim)
        # gamma_pdf = (x_values ** (shape_param - 1)) * np.exp(-x_values / scale_param) / (scale_param ** shape_param)
        # ax2.plot(x_values, gamma_pdf, 'r', label='Fitted gamma pdf')
    
    ax[0].set_title('H0 - all frames', y=1, pad=-14)
    ax[1].set_title('H1 - all frames', y=1, pad=-14)
    ax[2].set_title('H0 - videos avg', y=1, pad=-14)
    ax[3].set_title('H1 - videos avg', y=1, pad=-14)
    
    fig.suptitle(title, fontsize=16)
    plt.xticks(range(0, xlim + 1, 10))
    plt.savefig(os.path.join('plot', title))
    plt.cla()
    plt.clf()


def print_stats(A_pce_all, I_pce_all, A_pce_avg, title):
    with open(stats_txt, 'a') as f:
        IP_ratio = len(I_pce_all) / (len(A_pce_all) - len(I_pce_all)) if len(A_pce_all) > 0 else 0
        med_A = np.median(A_pce_all)
        med_I = np.median(I_pce_all)
        med_avg = np.median(A_pce_avg)
        f.write('-' * 8 + ' ' + title + ' ' + '-' * 8 + '\n')
        f.write(f'#I/#P frames:        {IP_ratio:10.4f}' + '\n')
        f.write(f'median all PCEs:     {med_A:10.4f}' + '\n')
        f.write(f'median I-frames:     {med_I:10.4f}' + '\n')
        f.write(f'median videos avg:   {med_avg:10.4f}' + '\n')
        if len(A_pce_all) > 0:
            f.write(f'95.0 percentile all: {np.percentile(A_pce_all, 95.0):10.4f}' + '\n')
            f.write(f'99.9 percentile all: {np.percentile(A_pce_all, 99.9):10.4f}' + '\n')
            f.write(f'max all frames:      {np.max(A_pce_all):10.4f}' + '\n')
            f.write(f'max videos avg:      {np.max(A_pce_avg):10.4f}' + '\n')
        
        def print_counts(sgn):
            aa = np.sum((A_pce_all > med_A) if sgn == '>' else (A_pce_all <= med_A))
            ii = np.sum((I_pce_all > med_A) if sgn == '>' else (I_pce_all <= med_A))
            pp = aa - ii
            ip = (ii / pp) / IP_ratio if ii > 0 else 0
            f.write(f'\nPCE {sgn} median_all_PCEs' + '\n')
            f.write('#I-frames:     %d' % ii + '\n')
            f.write('#P-frames:     %d' % pp + '\n')
            f.write('#I/#P normed:  %.4f' % ip + '\n')
            return ip
        
        ip0 = print_counts('<')
        ip1 = print_counts('>')
        f.write('\n\n')
        
        return med_A, med_I, med_avg, ip0, ip1


def main():
    global skip_flat_still
    folder_h0_icip = 'results/results_h0_icip'
    folder_h0_new = 'results/results_h0_new'
    folder_h0_new_2 = 'results/results_h0_new_2'
    
    folder_h0_no_inv = 'results/results_h0_no-inv'
    folder_h1_no_inv = 'results/results_h1_no-inv'
    folder_h0_no_inv_06_25 = 'results/results_h0_no-inv_06-25'
    folder_h1_no_inv_06_25 = 'results/results_h1_no-inv_06-25'
    
    folder_h1_icip = 'results/results_h1_icip'
    folder_h1_icip_I0 = 'results/results_h1_icip_I0'
    folder_h1_icip_GOP0 = 'results/results_h1_icip_GOP0'
    folder_h1_new = 'results/results_h1_new'
    folder_h1_new_2 = 'results/results_h1_new_2'
    folder_h1_new_I0 = 'results/results_h1_new_I0'
    folder_h1_new_GOP0 = 'results/results_h1_new_GOP0'
    folder_h1_raft = 'results/results_h1_raft'
    
    with open(stats_txt, 'w') as f:
        f.write('')
    
    for hist in [('icip', Method.ICIP),
                 ('new', Method.NEW),
                 ('new_2', Method.NEW),
                 ('no-inv', Method.NO_INV)]:
        plt_hist(*hist)
        plt_hist(*hist, devices=['D06', 'D25'])
    
    # plot_roc(folder_h0_icip, folder_h1_icip, 'SIFT skip', np.mean, True)
    # plot_roc(folder_h0_new, folder_h1_new, 'new skip', np.mean, True)
    devices = ['D06', 'D14', 'D15', 'D19', 'D25']
    devices = ['D06', 'D25']
    
    axss = plt_setup()
    
    # print(*get_avg_pce(folder_h1_new, np.mean, 'D25').astype(int))
    # print(*get_avg_pce(folder_h1_new_GOP0, np.mean, 'D25').astype(int))
    # print(*get_avg_pce(folder_h1_new_I0, np.mean, 'D25').astype(int))
    
    for i in range(2):
        axs = axss[:, i]
        plot_roc(axs, folder_h0_icip, folder_h1_icip, 'ICIP        ', np.mean, color=cm(4), lw=2)
        plot_roc(axs, folder_h0_icip, folder_h1_icip, 'ICIP thr=19 ', np.mean, color=cm(10), lw=2, tau=19)
        # plot_roc(axs, folder_h0_icip, folder_h1_icip_I0, 'SIFT --I_0  ', np.mean, color=cm(1), lw=2)
        # plot_roc(axs, folder_h0_icip, folder_h1_icip_GOP0, 'SIFT --GOP_0', np.mean, color=cm(5), lw=2)
        # plot_roc(axs, folder_h0_icip, folder_h1_raft, 'RAFT        ', np.mean, color=cm(6), lw=2)
        plot_roc(axs, folder_h0_new, folder_h1_new, 'NEW         ', np.mean, color=cm(8), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new, 'NEW         ', percentile_weights, color=cm(9), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new, 'NEW         ', power, color=cm(11), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new, 'NEW perc    ', percentile_weight, color=cm(1), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new, 'NEW max     ', np.max, color=cm(0), lw=2)
        # plot_roc(axs, folder_h0_test, folder_h1_test, 'no inv - ALL', np.mean, color=cm(0), lw=2)
        # plot_roc(axs, folder_h0_test, folder_h1_test, 'no inv - I', np.mean, frame_type='I', color=cm(6), lw=2)
        # plot_roc(axs, folder_h0_test, folder_h1_test, 'no inv - P', np.mean, frame_type='P', color=cm(2), lw=2)
        plot_roc(axs, folder_h0_no_inv, folder_h1_no_inv, 'no inv - ALL', np.mean, color=cm(1), lw=2)
        plot_roc(axs, folder_h0_no_inv, folder_h1_no_inv, 'no inv - I', np.mean, frame_type='I', color=cm(7), lw=2)
        plot_roc(axs, folder_h0_no_inv, folder_h1_no_inv, 'no inv - P', np.mean, frame_type='P', color=cm(3), lw=2)
        
        plot_roc(axs, folder_h0_no_inv_06_25, folder_h1_no_inv_06_25, 'no inv - ALL - 06 25', np.mean, color=cm(0), lw=2)
        plot_roc(axs, folder_h0_no_inv_06_25, folder_h1_no_inv_06_25, 'no inv - I - 06 25', np.mean, frame_type='I', color=cm(6), lw=2)
        plot_roc(axs, folder_h0_no_inv_06_25, folder_h1_no_inv_06_25, 'no inv - P - 06 25', np.mean, frame_type='P', color=cm(2), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new_I0, 'NEW --I_0   ', np.mean, color=cm(0), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new_GOP0, 'NEW --GOP_0 ', np.mean, color=cm(4), lw=2)
        # plot_roc(axs, folder_h0_new, folder_h1_new, 'SQUARE      ', np.mean, color=cm(8), lw=2)
        # for j, devide in enumerate(devices):
        #     plot_roc(axs, folder_h0_new, folder_h1_new, 'SIFT', np.mean, device=devide, lw=2, color=cm((j + 4) % 20), linestyle='--')
        #     plot_roc(axs, folder_h0_new, folder_h1_new, 'new', percentile_weight, device=devide, lw=2, color=cm((j + 4) % 20))
        
        axss[0, i].legend(loc='lower right')
        bar_labels = 'AUC, TPR\n%s'
        ticks = [bar_labels % 'ALL']
        for device in devices:
            ticks += [bar_labels % device]
        axss[1, i].set_xticks(range(len(devices) + 1), ticks)
        
        plot_roc.x = 0
        skip_flat_still = True
    
    plt.savefig('plot/roc.png')
    # plt.show()


if __name__ == '__main__':
    main()
