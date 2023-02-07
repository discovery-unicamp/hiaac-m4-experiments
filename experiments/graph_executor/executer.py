# Python imports
import numpy as np
import pandas as pd
import yaml
import networkx as nx

from pathlib import Path
from typing import Union

# Third-party imports

# Librep imports
from librep.datasets.multimodal.operations import *
from objects import the_objects


from networkx.drawing.nx_agraph import write_dot
from graphviz import Source

import warnings
warnings.filterwarnings('always')

labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

def load_yaml(path: Union[Path, str]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.CLoader)

class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.inputs = self.config["flow"].get("inputs", [])
        self.start_nodes = self.inputs  + list(self.config["flow"]["objects"].keys())
        self.graph = self.create_graph()
        self.objects = {}

    def get_obj_ref(self, obj):
        if isinstance(obj, str) and obj.startswith("$"):
            return self.objects[obj[1:]]
        return obj

    def get_node_name(self, obj):
        if isinstance(obj, str) and obj.startswith("$"):
            return obj[1:]
        return None

    def create_non_pipeline_objects(self):
        for key, value in self.config["flow"]["objects"].items():
            operation = the_objects[value["operation"]]
            init_args = [self.get_obj_ref(obj) for obj in value.get("args", [])]
            init_kwargs = {k: self.get_obj_ref(v) for k, v in value.get("kwargs", {}).items()}
            self.objects[key] = operation(*init_args, **init_kwargs)

    def create_graph(self):
        graph = nx.DiGraph()
        # graph.add_nodes_from(self.inputs)
        graph.add_nodes_from(self.start_nodes, s="^", b=1)
        for key, value in self.config["flow"]["objects"].items():
            graph.add_node(key)
            for node in [self.get_node_name(obj) for obj in value.get("args", [])]:
                if node:
                    graph.add_edge(node, key)
            for node in [self.get_node_name(v) for v in value.get("kwargs", {}).values()]:
                if node:
                    graph.add_edge(node, key)

        for key, value in self.config["flow"]["pipeline"].items():
            for node in [self.get_node_name(obj) for obj in value.get("init", {}).get("args", [])]:
                if node:
                    graph.add_edge(node, key)
            for node in [self.get_node_name(v) for v in value.get("init", {}).get("kwargs", {}).values()]:
                if node:
                    graph.add_edge(node, key)

            for node in [self.get_node_name(obj) for obj in value.get("call", {}).get("args", [])]:
                if node:
                    graph.add_edge(node, key)
            for node in [self.get_node_name(v) for v in value.get("call", {}).get("kwargs", {}).values()]:
                if node:
                    graph.add_edge(node, key)
        return graph

    def run(self, *args, **kwargs):
        for inp in self.inputs:
            self.objects[inp] = kwargs[inp]

        self.create_non_pipeline_objects()

        for x in nx.topological_sort(self.graph):
            print(x)

            if x in self.start_nodes:
                continue
                    
            obj = self.config["flow"]["pipeline"][x]
            init_args = [self.get_obj_ref(obj) for obj in obj.get("init", {}).get("args", [])]
            init_kwargs = {k: self.get_obj_ref(v) for k, v in obj.get("init", {}).get("kwargs", {}).items()}
            instance = the_objects[obj["operation"]](*init_args, **init_kwargs)

            call_args = [self.get_obj_ref(obj) for obj in obj.get("call", {}).get("args", [])]
            call_kwargs = {k: self.get_obj_ref(v) for k, v in obj.get("call", {}).get("kwargs", {}).items()}
            self.objects[x] = instance(*call_args, **call_kwargs)


def multimodal_dataframe_1():
    df = pd.DataFrame(
        np.arange(80).reshape(10, 8),
        columns=[
            "accel-0",
            "accel-1",
            "accel-2",
            "accel-3",
            "gyro-0",
            "gyro-1",
            "gyro-2",
            "gyro-3",
        ],
    )
    df["label"] = np.arange(10)
    return PandasMultiModalDataset(
        df, feature_prefixes=["accel", "gyro"], label_columns="label", as_array=True
    )


if __name__ == "__main__":
    x = load_yaml("/home/lopani/Documents/Doutorado/UNICAMP/H.IAAC-Meta4/hiaac-m4-experiments/experiments/graph_executor/simple_config.yaml")
    pipeline = Pipeline(x)
    print(pipeline)
    print(pipeline.graph)
    output = "/home/lopani/Documents/Doutorado/UNICAMP/H.IAAC-Meta4/hiaac-m4-experiments/experiments/graph_executor/simple_config.dot"
    write_dot(pipeline.graph, output)
    Source.from_file(output)
    train_dataset = multimodal_dataframe_1()
    test_dataset = multimodal_dataframe_1()
    reduce_dataset = multimodal_dataframe_1()

    pipeline.run(train_dataset=train_dataset, test_dataset=test_dataset, reduce_dataset=reduce_dataset, umap_dimensions=3)
