from random import randint

import numpy as np
from Client import Client
from Server import Server
from Network import Network
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def check_true(sv2, sv3, res):
    check_true = True
    for i in range(30):
        chbool = res[i] == sv2[i] + sv3[i]
        print(f"{sv2[i]} + {sv3[i]} = {res[i]} {chbool}")
        check_true = check_true and chbool

    if check_true:
        print("All True")
    else:
        print("There is false")
    

def main():
    # data = load_breast_cancer()
    # print(data.feature_names)
    X, y = load_breast_cancer(return_X_y=True)
    X1, y1 = X[0:135], y[0:135]
    X2, y2 = X[135:], y[135:]

    server = Server()
    client1 = Client()
    client2 = Client()

    server.set_secret_values([randint(1, 100) for i in range(32)])

    network = Network(server, [client1, client2], 3)
    server.set_network(network)
    
    client1.set_data(X1, y1)
    client2.set_data(X2, y2)

    server.initialize_model()
    print("Training on each client:")
    server.start_clients_training_and_update_models(1)

    print("\n*******************************")
    print("Evaluation on each client:")
    server.evaluate_each_model()

if __name__ == "__main__":
    main()