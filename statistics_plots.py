import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics_calc as sc
from scipy.stats import mode
from scipy.ndimage import label
import colorsys
import numpy as np
from matplotlib import colors, colorbar

sns.set_theme()

def _rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = colors.LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = colors.LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap


def color_labeled_pores(bin_img):
    fig, axes = plt.subplots(ncols=2, figsize=(14,7))
    axes[0].imshow(bin_img, cmap='gray')

    labels, _ = label(~bin_img)
    my_cmap = _rand_cmap(np.max(labels),
                         type='bright',
                         first_color_black=True,
                         last_color_black=False,
                         verbose=True)
    axes[1].imshow(labels, cmap=my_cmap)

    for ax in axes:
        ax.axis("off")

    return fig


def compare_stats(stats, names_of_stats, num_bins=50):
    nrows = 3
    ncols = len(stats)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), constrained_layout=True)

    if ncols==1:
        axes = [[ax] for ax in axes]

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
        size_percetile = np.percentile(stat, percetile)
        ax.axhline(0.9, color='red', linewidth=3, label=f"{percetile} percetile: x={size_percetile:.2f}")
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
