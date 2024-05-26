import pandas as pd
from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def tb_plot(fig_tag):
    log_dir = 'tb_logs'
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.histograms

    df = df[['step', f"{fig_tag}/counts", f"{fig_tag}/limits"]]
    df = df.dropna()

    # Set background
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Choose color palettes for the distributions
    pal = sns.color_palette("Oranges", 20)[5:-5]
    # Initialize the FacetGrid object (stacking multiple plots)
    g = sns.FacetGrid(df, row='step', hue='step', aspect=30, height=.4, palette=pal)

    def plot_subplots(x, color, label, data):
        ax = plt.gca()
        ax.text(0, .08, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
        counts = data[f"{fig_tag}/counts"].iloc[0]
        limits = data[f"{fig_tag}/limits"].iloc[0]
        x = np.linspace(limits[0], limits[-1], 15)
        x, y = SummaryReader.histogram_to_pdf(counts, limits, x)
        # Draw the densities in a few steps
        sns.lineplot(x=x, y=y, clip_on=False, color="w", lw=2)
        ax.fill_between(x, y, color=color)
    # Plot each subplots with df[df['step']==i]
    g.map_dataframe(plot_subplots, None)

    # Add a bottom line for each subplot
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    # Set the subplots to overlap (i.e., height of each distribution)
    g.figure.subplots_adjust(hspace=-.9)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], xlabel="", ylabel="")
    g.despine(bottom=True, left=True)

    g.savefig(f"figs/{fig_tag}.png")

tb_plot('Tanh seed#751 FC2')