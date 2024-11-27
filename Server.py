import flwr as fl

if __name__ == "__main__":
    # 启动Flower服务器，并监听指定的地址和端口
    strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 3 clients for training
    min_evaluate_clients=3,  # Never sample less than 3 clients for evaluation
    min_available_clients=3,  # Wait until all 3 clients are available
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",  # 监听所有 IP 地址上的 8080 端口
        config=fl.server.ServerConfig(num_rounds=10),  # 设置联邦学习的训练轮次
        strategy=strategy,
    )