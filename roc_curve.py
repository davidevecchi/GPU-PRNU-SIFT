import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import scipy.stats as stats
from sklearn.metrics import auc, roc_curve

from src.utils import Method

TARGET_FPR = 0.05

XMAX = .2
YMIN = 0.6
YTICKS = 9
XTICKS = 5

stats_txt = 'plot/stats.txt'
cm = plt.get_cmap('tab10')


def load_videos_pce(folder: str, func=np.mean, perc=95, devices=None):
    files = sorted(os.listdir(folder))
    mean_pce = []
    all_pce = np.empty(0, dtype=np.float32)
    for file in files:
        if file.endswith('.npz') and (devices is None or file[:3] in devices):  # os.path.basename(folder).startswith('H0') or
            # print(file)
            with np.load(os.path.join(folder, file)) as npz:
                pce_mat = npz['pce'].flatten()
            mean = func(pce_mat)
            mean_pce.append(mean)
            all_pce = np.concatenate([all_pce, pce_mat])
    if all_pce.size == 0:
        return [], 0, 0
    return np.array(mean_pce), np.percentile(all_pce, perc), np.max(all_pce)


def plot_roc__(axs, title, func=np.mean, devices=None, bar=False, **kwargs):
    h0 = f'results/H0_{title}'
    h1 = f'results/H1_{title}'
    if not (os.path.exists(h0) and os.path.exists(h1)):
        return 0
    pce_h0, perc_h0, max_h0 = load_videos_pce(h0, func, devices=devices)
    pce_h1, perc_h1, max_h1 = load_videos_pce(h1, func, devices=devices)
    print(f'{h0}  \t{perc_h0:8.4f}  {max_h0:10.4f}')
    print(f'{h1}  \t{perc_h1:8.4f}  {max_h1:10.4f}')
    print()
    
    # # fixme
    # step = 4
    # if frame_type == 'I':
    #     pce_h0 = pce_h0[0::step]
    #     pce_h1 = pce_h1[0::step]
    # elif frame_type == 'P':
    #     pce_h0 = [pce for i, pce in enumerate(pce_h0) if i % step != 0]
    #     pce_h1 = [pce for i, pce in enumerate(pce_h1) if i % step != 0]
    
    try:
        labels = [0] * len(pce_h0) + [1] * len(pce_h1)
        all_pce = np.concatenate([pce_h0, pce_h1])
        fpr, tpr, thresholds = roc_curve(labels, all_pce, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
    except ValueError as e:
        print(e)
        return 0
    
    tau_idx = (np.abs(fpr - TARGET_FPR)).argmin()
    tau = thresholds[tau_idx]
    
    if devices is None:
        devices = ''
    axs.plot(
            fpr, tpr, **kwargs,
            label='[AUC=%0.3f; TPR=%0.3f @ fpr=%0.4f, tau=%0.1f] %s' % (roc_auc, tpr[tau_idx], fpr[tau_idx], tau, title + ' ' + str(devices))
    )
    
    # width = 0.3
    # gap = 0.05
    # multiplier = [-1, 1]
    # new = plot_roc.x > int(plot_roc.x)
    #
    # def formatt(val):
    #     return ('%.3f' % val)[1:] if val < .9999 else '1.00'
    #
    # if not bar:
    #     return
    #
    # for i, y, l in zip(multiplier, [roc_auc, tpr[tau_idx]], ['auc', 'tpr']):
    #     offset = (width + gap) * i * 0.5
    #     if new:
    #         kwargs['color'] = cm(1)
    #         kwargs['zorder'] = -1
    #     rects = axs[1].bar(int(plot_roc.x) + offset, y, width, label=l, **kwargs)
    #     axs[1].bar_label(rects, labels=[formatt(y)], padding=3 if new else -15)
    # plot_roc.x += 0.5
    
    return 1


def plot_roc(axs, experiments, mode=None, devices=None):
    pce_h0_all, pce_h1_all = [], []
    modes = ['ALL', 'I0', 'GOP0']
    
    for e in experiments:
        for m in modes if mode is None else [mode]:
            h0 = f'results/H0_{e}_{m}'
            h1 = f'results/H1_{e}_{m}'
            if not (os.path.exists(h0) and os.path.exists(h1)):
                return None
            pce_h0, perc_h0, max_h0 = load_videos_pce(h0, devices=devices)
            pce_h1, perc_h1, max_h1 = load_videos_pce(h1, devices=devices)
            pce_h0_all += pce_h0.tolist()
            pce_h1_all += pce_h1.tolist()
    
    try:
        labels = [0] * len(pce_h0_all) + [1] * len(pce_h1_all)
        all_pce = np.concatenate([pce_h0_all, pce_h1_all])
        fpr, tpr, thresholds = roc_curve(labels, all_pce, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
    except ValueError as e:
        print(e)
        return None
    
    tau_idx = np.argmin(np.abs(fpr - TARGET_FPR))
    if tau_idx == 0:
        tau_idx = np.argmin(np.abs(fpr - TARGET_FPR - 0.01))
    
    title = experiments[0] if len(experiments) == 1 else experiments[0][:3]  # fixme
    
    if mode is None:
        mode = 'AVG'
        j = 0
    else:
        j = modes.index(mode)
    plot_roc.experiments.add(title)
    color = cm(len(plot_roc.experiments) - 1)
    
    tpr_ = tpr if devices is None else tpr - (0.003 * (len(plot_roc.experiments) - 1))  # fixme
    axs.plot(fpr, tpr_, color=color, lw=1.5, linestyle=['-', '--', ':'][j])
    numbers = '%0.3f %0.3f %0.1f' % (roc_auc, tpr[tau_idx], thresholds[tau_idx])
    r = [title, mode, *numbers.split(' ')]
    rows.append(r)
    markers.append(['━━━━', '━ ━ ━', '╍╍╍╍'][j])
    colors.append(color)


plot_roc.x = 0
plot_roc.experiments = set()


def plt_setup(xmax=XMAX, ymin=YMIN, xticks=XTICKS, yticks=YTICKS):
    fig, axs = plt.subplots(tight_layout=True, figsize=(8, 8), dpi=300)  # , sharey='row')
    
    # fig = plt.figure(constrained_layout=True, figsize=(16, 16), dpi=300)
    # subfigs = fig.subfigures(nrows=2, ncols=1)
    # axs = np.empty((2, 2), dtype=object)
    # titles = ['Receiver Operating Characteristic', 'AUC / TPR comparison']
    # for i, subfig in enumerate(subfigs):
    #     subfig.suptitle(titles[i])
    #     ax = subfig.subplots(nrows=1, ncols=2)  # , sharey='row'
    #     axs[i, 0], axs[i, 1] = ax[0], ax[1]
    
    # axc = [None, None]
    # for i in range(2):
    axs.plot([TARGET_FPR, TARGET_FPR], [0.0, 1.0], lw=0.75, linestyle='--', color='gray')
    axs.plot([0, 1], [1.0, 1.0], lw=0.5, linestyle='--', color='gray')
    axs.set_xlim([0.0, xmax])
    axs.set_ylim([ymin, 1])
    axs.set_xticks(np.linspace(0, xmax, xticks))
    axs.set_yticks(np.linspace(ymin, 1, yticks))
    axs.set_xlabel('False Positive Rate')
    
    # axs.grid(visible=True, axis='y', lw=0.5, linestyle='--')
    # axs.set_ylim([ymin, 1])
    # axs.set_yticks(np.linspace(ymin, 1, TICKS))
    # axs.set_xlabel('Metrics, Devices')
    
    # fixme: add fake label '1.0' at the SE corner of the second row, to match the width of the first row
    # axc[i] = axs[1, i].twiny()
    # axc[i].set_xticks([1, 1.1])
    # axc[i].set_xlim([0, 1])
    # axc[i].tick_params(bottom=False, top=False, labelbottom=True, labeltop=False, labelcolor='w')
    
    axs.set_ylabel('True Positive Rate')
    # axs[1, 0].set_ylabel('AUC / TPR @ fpr=0.05')
    
    return axs


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
                # elif method == Method.NO_INV:
                #     I_pce_vid = A_pce_vid[0::4]
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
        
        if i == 0:
            ax2 = ax[i + 0].twinx()
            shape, loc, scale = stats.gamma.fit(A_pce_all)  # [A_pce_all > 17])  # fixme
            gamma_x = np.linspace(0, xlim, 10 * xlim)
            gamma_y = stats.gamma.pdf(gamma_x, shape, loc, scale)
            ax2.plot(gamma_x, gamma_y, color='r')
            ax2.set_ylim([0, np.max(gamma_y) * 1.08])
    
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


rows = []
markers = []
colors = []


def main():
    with open(stats_txt, 'w') as f:
        f.write('')
    
    # devices = ['D06', 'D14', 'D15', 'D19', 'D25']
    devices = [None, None, ['D06'], ['D25']]
    xmaxs = [0.2, 1, 1, 1]
    ymins = [0.6, 0.6, 0, 0]
    xticks = [5, 5, 5, 5]
    yticks = [9, 9, 11, 11]
    
    for device, xmax, ymin, xt, yt in zip(devices, xmaxs, ymins, xticks, yticks):
        axss = plt_setup(xmax, ymin, xt, yt)
        
        rows.clear()
        markers.clear()
        colors.clear()
        plot_roc.experiments.clear()
        xy_int = xmax == 1
        
        for method in ['ICIP', 'NEW']:
            modes = [None] if (device is None and xy_int) else ['ALL', 'I0', 'GOP0']
            for mode in modes:
                plot_roc(axss, experiments=[method], mode=mode, devices=device)
        plot_roc(axss, experiments=['RAFT'], devices=device)
        plot_roc(axss, experiments=['RND', 'RND0', 'RND1', 'RND2', 'RND3'], devices=device)
        
        collabel = ('Method', 'Mode', 'AUC', 'TPR', 'τ')
        table = axss.table(cellText=rows, colLabels=collabel,
                           loc='center', cellLoc='center',
                           colWidths=[0.1] * len(rows[0]),
                           bbox=[0.55, 0.01, .44, .032 * (len(rows) + 1)])
        height = table.get_celld()[0, 0].get_height()
        for i in range(len(rows)):
            cell = table.add_cell(i + 1, -2, width=0.1, height=height, text=markers[i], loc='center')
            cell.get_text().set_color(colors[i])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # bar_labels = 'AUC, TPR\n%s'
        # ticks = [bar_labels % 'ALL']
        # for device in devices:
        #     ticks += [bar_labels % device]
        # axss[1, a].set_xticks(range(len(devices) + 1), ticks)
        
        dev = 'all' if device is None else '-'.join(device)
        if not xy_int:
            dev += f'_{xmax}-{ymin}'
        plt.savefig(f'plot/roc_{dev}.png')
        plt.show()


if __name__ == '__main__':
    main()
