import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report

from .data import language_names, small_languages
from .labels import binarize_labels, get_label_scheme, map_full_names


class Clustering:
    a = 1
