from anemoi.utils.config import DotDict


class AnemoiGraphSchema:
    def __init__(self, graph_data: dict, config: DotDict) -> None:
        self.hidden_name = config.graphs.hidden_mesh.name
        self.mesh_names = [name for name in graph_data if isinstance(name, str)]
        self.input_meshes = [
            k[0] for k in graph_data if isinstance(k, tuple) and k[2] == self.hidden_name and k[2] != k[0]
        ]
        self.output_meshes = [
            k[2] for k in graph_data if isinstance(k, tuple) and k[0] == self.hidden_name and k[2] != k[0]
        ]
        self.num_nodes = {name: graph_data[name]["coords"].shape[0] for name in self.mesh_names}
        self.num_node_features = {name: 2 * graph_data[name]["coords"].shape[1] for name in self.mesh_names}
        self.num_trainable_params = {
            name: config.model.trainable_parameters["data" if name != "hidden" else name]
            for name in self.mesh_names
        }

    def get_node_emb_size(self, name: str) -> int:
        return self.num_node_features[name] + self.num_trainable_params[name]
