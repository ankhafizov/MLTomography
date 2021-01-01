import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics_calc as sc
from scipy.stats import mode

sns.set_theme()

def compare_stats(stats, names_of_stats, num_bins=50):
    nrows = 3
    ncols = len(stats)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), constrained_layout=True)

    for ax, stat, title in zip(axes[0], stats, names_of_stats):
        sns.histplot(stat, bins=num_bins, ax=ax).set_title(title+f" std: {np.std(stat):.2f}")

        mean = np.mean(stat)
        ax.axvline(mean, color='red', linewidth=3, label=f"mean: {mean:.2f}")

        mode_value = int(mode(stat)[0])
        ax.axvline(mode_value, color='purple', linewidth=2, label=f"mode: {mode_value:.2f}")
        # median = np.median(stat)
        # ax.axvline(median, color='green', linewidth=3, label=f"median: {median:.2f}")
        ax.legend()

    for ax, stat, title in zip(axes[1], stats, names_of_stats):
        sns.histplot(stat, bins=num_bins, cumulative=True, kde=True, ax=ax).set_title(f'cumulative {title}')

    for ax, stat, title in zip(axes[2], stats, names_of_stats):
        sns.histplot(stat, 
                     bins=num_bins,
                     cumulative=True,
                     kde=True,
                     stat="density",
                     ax=ax).set_title(f'cumulative density {title}')

        percetile = 90
        length_percetile = np.percentile(stat, percetile)
        ax.axhline(0.9, color='red', linewidth=3, label=f"{percetile} percetile: x={length_percetile}")
        ax.axvline(length_percetile, color='red', linewidth=3)
        ax.legend(loc=4)

    return fig


def show_phantom_section_and_profile(phantom, num_row, profile_orientation='v'):
    fig, axes = plt.subplots(nrows=2, figsize=(5,10))
    num_row = 300
    axes[0].imshow(phantom, cmap='gray')
    axes[0].axis("off")

    if profile_orientation == 'h':
        axes[1].plot(phantom[num_row])
        axes[0].axhline(num_row, color='red', linewidth=3)
    elif profile_orientation == 'v':
        axes[1].plot(phantom[:, num_row])
        axes[0].axvline(num_row, color='red', linewidth=3)

    axes[1].set_xlim(xmin=0, xmax=len(phantom[num_row]))
    plt.tight_layout()

    return fig


def plot_data_scatter(df, add_regression_plane=False):
    train_data = sc.extract_data_from_dataframe(df)

    porosity_min, blobiness_min = np.min(train_data[0]), np.min(train_data[1])
    porosity_max, blobiness_max = np.max(train_data[0]), np.max(train_data[1])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*train_data[0:-1], train_data[-1], color='red', linewidth=3)

    ax.set_xlabel('porosity')
    ax.set_ylabel('blobiness')
    ax.set_zlabel('hist mean')

    if add_regression_plane:
        p, b = np.meshgrid([porosity_min, porosity_max], [blobiness_min, blobiness_max])
        intercept, p_coef, b_coef = sc.get_regression_coefs_by_plane(df)
        hist_mean = intercept + p*p_coef + b*b_coef
        ax.plot_surface(p, b, hist_mean, alpha=0.3)

    return fig
