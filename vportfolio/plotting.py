# Functions related to plotting

from matplotlib.ticker import LogLocator, StrMethodFormatter, NullLocator

# I decided to make a function with a method found on stack overflow
# it will be called when plotting and will adjust y ticks for log scales
def format_log_axis(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5])) # ticks at 1,2 and 5 times every power of 10
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}")) # writes each tick in plain number
    ax.yaxis.set_minor_locator(NullLocator()) # removes unlabelled minor ticks    