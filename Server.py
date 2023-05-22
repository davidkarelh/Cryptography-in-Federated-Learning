import math
import numpy as np
from Network import Network
from Node import Node
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Server(Node):
    _network: Network
    _model: LogisticRegression
    MODEL_RANDOM_STATE = 0
    MODEL_SOLVER = "liblinear"

    def __init__(self) -> None:
        super().__init__()
    
    def set_network(self, network: Network):
        self._network = network
    
    def coordinate_secret_aggregatting(self):
        intermediate_results = self.coordinate_results_sending()
        final_results = self._calculate_final_results(intermediate_results)
        return final_results
    
    def _calculate_final_results(self, intermediate_results: list[list[float]]):
        ret: list[float] = []
        clients_nodes = self._network.get_clients_nodes()

        a: list[list[float]] = []
        b: list[list[float]] = [[] for i in range(len(intermediate_results[0]))]
        a_idx = 0

        a.append([])
        for i in range(self._network.get_minimum_number_of_colluding_parties()):
            a[a_idx].append(math.pow(self.public_value, i))

        a_idx += 1

        for clients_node in clients_nodes:
            if a_idx >= self._network.get_minimum_number_of_colluding_parties():
                break

            a.append([])
            for i in range(self._network.get_minimum_number_of_colluding_parties()):
                a[a_idx].append(math.pow(clients_node.public_value, i))
            a_idx += 1

        for intermediate_results_each_node in intermediate_results:
            for idx, intermediate_result_each_value in enumerate(intermediate_results_each_node):
                b[idx].append(intermediate_result_each_value)


        for idx, b_val in enumerate(b):
            # print("linalg")
            # print(a)
            # print(b_val[0:self._network.get_minimum_number_of_colluding_parties()])
            solved = np.linalg.solve(a, b_val[0:self._network.get_minimum_number_of_colluding_parties()])
            ret.append(solved[0] - self._secret_values[idx])
            # print(solved)

        # print("ret")
        # print(ret)
        return ret
    
    def coordinate_results_sending(self):
        clients_nodes = self._network.get_clients_nodes()
        results: list[list[list[float]]] = [[] for i in range(1 + len(clients_nodes))]
        ret: list[list[float]] = []
        # To server
        results[0].append(self.calculate_result(self.public_value))
        for client_node in clients_nodes:
            results[0].append(client_node.calculate_result(self.public_value))
        
        # To clients
        for i in range(1, len(clients_nodes) + 1):
            results[i].append(self.calculate_result(clients_nodes[i - 1].public_value))

            for idx, client_node in enumerate(clients_nodes):
                results[i].append(client_node.calculate_result(clients_nodes[i - 1].public_value))

        for idx, results_each_node in enumerate(results):
            if idx == 0:
                ret.append(self.aggregate_to_get_intermediate_results(results_each_node))
            else:
                ret.append(clients_nodes[idx - 1].aggregate_to_get_intermediate_results(results_each_node))
        
        return ret

    def initialize_model(self):
        self._model = LogisticRegression(random_state=Server.MODEL_RANDOM_STATE, solver=Server.MODEL_SOLVER)
        clients_nodes = self._network.get_clients_nodes()

        for clients_node in clients_nodes:
            clients_node.initialize_model(Server.MODEL_RANDOM_STATE, Server.MODEL_SOLVER)
            if hasattr(self._model, "coef_"):
                clients_node.set_coef_and_intercept(self._model.coef_)
        
    def start_clients_training_and_update_models(self, num_of_rounds: int):
        clients_nodes = self._network.get_clients_nodes()
        for i in range(num_of_rounds):
            print(f"\nRound {i + 1}")
            clients_coefs = []
            clients_data_lengths = []
            clients_data_intercepts = []

            for idx, clients_node in enumerate(clients_nodes):
                print(f"Client {idx + 1} training:")
                a, b, intrcpt = clients_node.train_and_return_weights()
                clients_coefs.append(a)
                clients_data_lengths.append(b)
                clients_data_intercepts.append(intrcpt[0])

            secure_aggregate_result = self.coordinate_secret_aggregatting()
            num_total_of_data = secure_aggregate_result[0]
            weighted_sum_intercept = secure_aggregate_result[1]
            weighted_sum = secure_aggregate_result[2:]

            weighted_sum /= num_total_of_data
            weighted_sum_intercept /= num_total_of_data

            if hasattr(self._model, "coef_"):
                self._model.coef_ = self._model.coef_ + weighted_sum
                self._model.intercept_[0] = self._model.intercept_[0] + weighted_sum_intercept

            else:
                self._model.coef_ = np.array([weighted_sum])
                self._model.intercept_ = np.array([weighted_sum_intercept])
                
            for clients_node in clients_nodes:
                clients_node.set_coef_and_intercept(self._model.coef_, self._model.intercept_) 

    def evaluate_each_model(self):
        clients_nodes = self._network.get_clients_nodes()
        for idx, clients_node in enumerate(clients_nodes):
            print(f"Client {idx + 1} evaluation:")
            clients_node.evaluate_on_local_data()

    def get_model(self):
        return self._model

