from typing import List, Union
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from librep.utils.visualization.multimodal_har import plot_windows_sample


def compare_metadata(dataset_normal, dataset_resampled, columns: Union[str, List[str]]):
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column == "index":
            # They all should result in the same fraction if they are linearly dependent
            if len((dataset_normal[column] / dataset_resampled[column]).unique()) != 1:
                print(f"Columns '{column}' are not equal")
                return False

        elif not np.all(dataset_normal[column] == dataset_resampled[column]):
            print(f"Columns '{column}' are not equal")
            return False
    return True


def generate_plots(
    *datasets,
    sample_no: int = 0,
    the_slice: slice = slice(None, None, None),
    windows: List[str] = None,
    height: int = 600,
    width: int = 800,
    names: List[str] = None,
    vertical_spacing: float = 0.1,
    title: str = "",
    x_title: str = "x",
    y_title: str = "y",
):
    color10_16 = ["blue", "cyan", "magenta", "#636efa", "#00cc96", "#EF553B", "brown"]
    if windows is None:
        windows = datasets[0].window_names

    if names is None:
        names = [f"Dataset {i}" for i in range(len(datasets))]

    fig = make_subplots(
        rows=len(datasets),
        cols=1,
        start_cell="top-left",
        subplot_titles=names,
        vertical_spacing=vertical_spacing,
        x_title=x_title,
        y_title=y_title,
    )

    for i, dataset in enumerate(datasets):
        traces, _ = plot_windows_sample(
            dataset,
            windows=windows,
            sample_idx=sample_no,
            the_slice=the_slice,
            title=f"Dataset {i}",
            return_traces_layout=True,
        )
        for j, trace in enumerate(traces):
            trace.line.color = color10_16[j]
            trace.legendgroup = str(j)
            trace.showlegend = False if i > 0 else True

        fig.add_traces(traces, rows=[i + 1] * len(traces), cols=[1] * len(traces))

    fig.update_layout(title=title)
    fig.update_layout(height=height, width=width)

    return fig
