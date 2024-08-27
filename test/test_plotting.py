import matplotlib.pyplot as plt
from pathlib import Path

from test.config import TEST_DIR
from util.plotting import plot_logs


def manual_test_plot():
    path = Path(f"{TEST_DIR}/resource/logs/")
    plot_logs(path)
    plt.show()
