# test functions for plotting.py

import matplotlib
matplotlib.use("Agg") # draw without a screen (needed in CI)
import matplotlib.pyplot as plt
from vportfolio.plotting import format_log_axis

# format_log_axis
def test_format_log_axis():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 4])
    ax.set_yscale("log")
    format_log_axis(ax)
    # labels are plain numbers, and there are no minor ticks
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert "1" in labels and "2" in labels
    assert len(ax.yaxis.get_minorticklocs()) == 0