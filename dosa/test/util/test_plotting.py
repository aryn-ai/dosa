import matplotlib.pyplot as plt
from pathlib import Path

from dosa.test.config import TEST_DIR
from dosa.util.plotting import plot_logs


def manual_test_plot():
    path = Path(f"{TEST_DIR}/resource/logs/")
    plot_logs(path)
    plt.show()
