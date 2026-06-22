import json
import os

class ModelTree:
    filename = 'model_metadata.json'

    def __init__(self, data_dir, task):
        """
        Initializes an empty ModelMetadata object.
        """
        self.adj_list = {}
        self.node_metadata = {}  # Add this to store metadata for each node
        self.edge_types = ('episode', 'weight')
        self.data_dir = data_dir
        self.task = task
        

    def add_new_version(self, version, metadata=None):
        """
        Adds a new node if it doesn't already exist, with optional metadata.

        Args:
            version (str): The unique identifier for the node.
            metadata (dict, optional): Metadata for the node.
        """
        if version not in self.adj_list:
            self.adj_list[version] = {self.edge_types[0]: [], self.edge_types[1]: []}
            self.node_metadata[version] = metadata if metadata is not None else {}

    def _add_node(self, version: str, metadata: dict = None):
        # Alias for add_new_version for internal use
        self.add_new_version(version, metadata)

    def set_node_metadata(self, version: str, metadata: dict = None):
        """
        Sets or updates metadata for a node.

        Args:
            version (str): Node identifier.
            metadata (dict): Metadata to set.
        """
        if version not in self.adj_list:
            self.add_new_version(version, metadata)
        else:
            self.node_metadata[version] = metadata

    def get_node_metadata(self, version: str):
        """
        Gets metadata for a node.

        Args:
            version (str): Node identifier.

        Returns:
            dict: Metadata for the node, or None if not found.
        """
        return self.node_metadata.get(version, None)

    def _add_edge(self, base_version: str, new_version: str, edge_type: str):
        """
        A private helper method to add a directed edge between two nodes.

        Args:
            new_version (str): The identifier of the new node.
            base_version (str): The identifier of the base node.
            edge_type (str): The type of the edge ('episode' or 'weight').
        """
        # Ensure both nodes exist in the metadata structure
        self._add_node(new_version)
        self._add_node(base_version)

        # Add the edge to the source node's list for the given edge type
        if base_version not in self.adj_list[new_version][edge_type]:
            self.adj_list[new_version][edge_type].append(base_version)

    def add_episode_edge(self, base_version: str, new_version: str):
        """
        Adds a directed 'episode' edge between two nodes.

        Args:
            base_version (str): The identifier of the base node.
            dest_id (str): The identifier of the destination node.
        """
        self._add_edge(base_version, new_version, 'episode')

    def add_weight_edge(self, base_version: str, new_version: str):
        """
        Adds a directed 'weight' edge between two nodes.

        Args:
            base_version (str): The identifier of the base node.
            new_version (str): The identifier of the destination node.
        """
        self._add_edge(base_version, new_version, 'weight')

    def __str__(self):
        """
        Provides a string representation of the metadata for printing.
        """
        if not self.adj_list:
            return "ModelMetadata is empty."
            
        output = "ModelMetadata Structure:\n"
        for node, connections in self.adj_list.items():
            output += f"- Node '{node}':\n"
            meta = self.node_metadata.get(node, None)
            output += f"  - metadata: {meta}\n"
            for edge_type, neighbors in connections.items():
                if neighbors:
                    output += f"  - {edge_type} -> {', '.join(neighbors)}\n"
                else:
                    output += f"  - {edge_type} -> [None]\n"
        return output

    def save_to_disk(self):
        """
        Saves the metadata's adjacency list and node metadata to a file in JSON format.
        """
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Data directory '{self.data_dir}' cannot be found.")
        file = os.path.join(self.data_dir, self.task, self.filename)
        try:
            with open(file, 'w') as f:
                json.dump({
                    'adj_list': self.adj_list,
                    'node_metadata': self.node_metadata
                }, f, indent=4)
            print(f"ModelMetadata successfully saved to '{file}'.")
        except IOError as e:
            print(f"Error saving metadata to file: {e}")

    @classmethod
    def load_from_disk(cls, data_dir, task):
        """
        Loads metadata from a JSON file.

        Args:
            data_dir (str): The directory where the metadata file is located.

        Returns:
            ModelMetadata: A new ModelMetadata object instance with the loaded data.
                           Returns None if the file cannot be read or parsed.
        """
        file = os.path.join(data_dir, task, cls.filename)
        try:
            with open(file, 'r') as f:
                loaded = json.load(f)
            new_metadata = cls(data_dir, task)
            new_metadata.adj_list = loaded.get('adj_list', {})
            new_metadata.node_metadata = loaded.get('node_metadata', {})
            return new_metadata
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            return None
        
    
    def get_weight_edges(self, version: str):
        """
        Gets all weight edges for a given version.

        Args:
            version (str): The identifier of the node.

        Returns:
            list: A list of versions that are connected by weight edges.
        """
        if version not in self.adj_list:
            return []
        return self.adj_list[version]['weight']
    

    def get_episode_edges(self, version: str):
        """
        Gets all episode edges for a given version.

        Args:
            version (str): The identifier of the node.

        Returns:
            list: A list of versions that are connected by episode edges.
        """
        if version not in self.adj_list:
            return []
        return self.adj_list[version]['episode']
    
        
    def _walk_back(self, version: str, edge_type: str = 'episode', lookback: int = 1):
        """
        Walks backward through the metadata structure from a given version.

        Args:
            version (str): The starting node identifier.
            edge_type (str): The type of edges to follow ('episode' or 'weight').
            lookback (int): The number of steps to walk back.

        Returns:
            list: A list of versions that can be reached by following the specified edge type.
        """
        if version not in self.adj_list:
            return []

        result = {}
        cur_adj_list = self.adj_list[version][edge_type]
        for adj in cur_adj_list:
            if len(result) >= lookback:
                break
            result[adj] = True

        for adj in cur_adj_list:
            if len(result) >= lookback:
                break
            next_result = self._walk_back(adj, edge_type, lookback - len(result))
            result.update(next_result)

        return list(result.keys())


    def walk_back_episode(self, version: str, lookback: int = 1):
        """
        Walks backward through the metadata structure from a given version using 'episode' edges.

        Args:
            version (str): The starting node identifier.
            lookback (int): The number of steps to walk back.

        Returns:
            list: A list of versions that can be reached by following 'episode' edges.
        """
        return self._walk_back(version, edge_type='episode', lookback=lookback)

    def walk_back_weight(self, version: str, lookback: int = 1):
        """
        Walks backward through the metadata structure from a given version using 'weight' edges.

        Args:
            version (str): The starting node identifier.
            lookback (int): The number of steps to walk back.

        Returns:
            list: A list of versions that can be reached by following 'weight' edges.
        """
        return self._walk_back(version, edge_type='weight', lookback=lookback)


# if __name__ == "__main__":
#     # Example usage
#     metadata = ModelTree(data_dir='/media/vivekbagade/Elements/act-data')
#     metadata.add_new_version('1.0.0', {'description': 'Initial version'})
#     metadata.add_new_version('1.1.0', {'description': 'Finetuning test'})
#     metadata.add_new_version('1.2.0', {'description': 'Can in left far area'})

#     metadata.add_episode_edge(base_version='1.0.0', new_version='1.1.0')
#     metadata.add_weight_edge(base_version='1.0.0', new_version='1.1.0')

#     metadata.add_episode_edge(base_version='1.0.0', new_version='1.2.0')
#     metadata.add_weight_edge(base_version='1.0.0', new_version='1.2.0')

#     print(metadata)
    
#     # Save to disk
#     metadata.save_to_disk()
    
#     # Load from disk
#     loaded_metadata = ModelTree.load_from_disk('/media/vivekbagade/Elements/act-data')
#     print(loaded_metadata)
