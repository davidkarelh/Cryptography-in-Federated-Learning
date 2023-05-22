from Client import Client
from Node import Node

class Network:
    _server_node: Node
    _clients_nodes: list[Client]
    _minimum_number_of_colluding_parties: int = 0

    def __init__(self, server: Node, clients_nodes: list[Client], minimum_number_of_colluding_parties: int) -> None:
        self._server_node = server
        self._clients_nodes = clients_nodes
        self._minimum_number_of_colluding_parties = minimum_number_of_colluding_parties

        for clients_node in self._clients_nodes:
            clients_node.set_server(self._server_node)

        self._initialize_nodes_values()
    
    def get_clients_nodes(self):
        return self._clients_nodes

    def get_minimum_number_of_colluding_parties(self):
        return self._minimum_number_of_colluding_parties
    
    def _initialize_nodes_values(self):
        self._server_node.generate_secret_coefficients(self._minimum_number_of_colluding_parties)
        self._server_node.generate_public_value()

        for clients_node in self._clients_nodes:
            clients_node.generate_secret_coefficients(self._minimum_number_of_colluding_parties)
            clients_node.generate_public_value()