import numpy as np
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go

from .labels import get_label_scheme, binarize_labels, map_full_names
from .data import small_languages, language_names


class Clustering:
    a = 1
