import matplotlib.pyplot as plt

from matplotlib import patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib.colors as colors

cmap = cm.get_cmap('viridis')
cmap_r = cm.get_cmap('viridis_r')

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'xtick.direction':'in'})
matplotlib.rcParams.update({'ytick.direction':'in'})
matplotlib.rcParams.update({'xtick.top':True})
matplotlib.rcParams.update({'ytick.right':True})
matplotlib.rcParams.update({'legend.frameon':False})
matplotlib.rcParams.update({'lines.dashed_pattern':[5,3]})
