import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(
    logs,
    ewm_col=0,
    log_name="log.txt",
):
    """
    Function to plot specific fields from training log(s). Plots both training
    and test results. matplotlib plots of results in fields, color coded for
    each log file. - solid lines are training results, dashed lines are test
    results.

    :param logs: list containing Path objects, each pointing to individual dir
        with a log file
    :param ewm_col: optional, which column to use as the exponential weighted
        smoothing of the plots
    :param log_name: optional, name of log file, default 'log.txt'.
    """
    func_name = "plotting.py::plot_logs"
    fields = ("loss", "parent")

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
        else:
            raise ValueError(f"{func_name} - invalid argument {logs}.")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument: {dir}"
            )
        if not dir.exists():
            raise ValueError(
                f"{func_name} - invalid directory in logs argument: {dir}"
            )

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))
    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            df.interpolate().ewm(com=ewm_col).mean().plot(
                y=[f"train_{field}", f"test_{field}"],
                ax=axs[j],
                color=[color] * 2,
                style=["-", "--"],
            )
    for ax, field in zip(axs, fields):
        ax.legend(["train", "test"])
        ax.set_title(field)
