import flwr as fl

def main():
    # Configure the server (e.g., 1 federated round)
    server_config = fl.server.ServerConfig(num_rounds=3)

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=1,  # Set to 1
        min_fit_clients=1,        # Add this parameter 
        min_evaluate_clients=1    # Add this parameter
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config
    )

if __name__ == "__main__":
    main()