import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    times = {
        'ICIP': [[9, 23], [18, 21]],
        'RAFT': [[7, 21], [13, 10]],
        'NEW': [[2, 10], [3, 53]]
    }
    
    all = [t[0][0] + t[0][1] / 60 for t in times.values()]
    others = [(t[1][0] + t[1][1] / 60) / 2 for t in times.values()]
    
    X_axis = np.arange(len(times))
    
    plt.bar(X_axis, all, 0.75, label='All computations', color=['tab:blue', 'tab:green', 'tab:orange'])
    # plt.bar(X_axis + 0.15, others, 0.3, label='Pre-computed')
    plt.ylim([0, 10])
    plt.xlim(-0.6, len(times) - 0.4)
    plt.xticks(X_axis, times.keys())
    plt.title("Computational cost")
    plt.ylabel("Execution time (h)")
    plt.tight_layout()
    plt.savefig('plot/time.pdf')
    plt.show()
