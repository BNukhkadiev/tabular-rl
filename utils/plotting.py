# utils/plotting.py

import matplotlib.pyplot as plt

def plot_convergence(curves, labels, title, xlabel, ylabel, filepath=None):
    """
    Plot one or more convergence curves on the same axes.

    Args:
        curves:   list of 1D iterables (e.g. list of deltas per iteration)
        labels:   list of strings for each curve
        title:    plot title
        xlabel:   x-axis label
        ylabel:   y-axis label
        filepath: if given, save figure to this path
    """
    for curve, lab in zip(curves, labels):
        plt.plot(curve, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()
