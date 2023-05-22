import math
from random import randint


class Node:
    MIN_SECRET_COEF = 1
    MAX_SECRET_COEF = 100
    MIN_PUBLIC_VALUE = 1
    MAX_PUBLIC_VALUE = 100

    _secret_coefficients: list[int]
    _secret_values: list[float]
    public_value: int

    def __init__(self) -> None:
        pass

    def generate_secret_coefficients(self, num_of_colluding_parties: int):
        self._secret_coefficients = []
        for i in range(num_of_colluding_parties - 1):
            self._secret_coefficients.append(randint(Node.MIN_SECRET_COEF, Node.MAX_SECRET_COEF))
            # self._secret_coefficients.append(Node.MIN_SECRET_COEF)
    
    def generate_public_value(self):
        self.public_value = randint(Node.MIN_PUBLIC_VALUE, Node.MAX_PUBLIC_VALUE)
        # self.public_value = Node.MAX_PUBLIC_VALUE
    
    def set_secret_values(self, secret_values: list[float]):
        self._secret_values = secret_values

    def calculate_result(self, destination_public_value: int):
        ret: list[float] = []

        for secret_value in self._secret_values:
            result = 0
            for idx, secret_coefficient in enumerate(self._secret_coefficients):
                result += (secret_coefficient * math.pow(destination_public_value, idx + 1))
                # print(idx)
            # print(result)
            # print(secret_value)
            result += secret_value
            ret.append(result)
        
        return ret

    def aggregate_to_get_intermediate_results(self, received_results: list[list[float]]):
        ret: list[float] = [0 for i in range(len(received_results[0]))]
        for received_result in received_results:
            for idx, rec in enumerate(received_result):
                ret[idx] += rec
        
        return ret