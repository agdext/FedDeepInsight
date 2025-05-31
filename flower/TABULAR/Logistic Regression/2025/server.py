import flwr as fl
import utils
from sklearn.linear_model import LogisticRegression
from flwr.server.strategy import DifferentialPrivacyServerSideAdaptiveClipping





# Start Flower server for x rounds of federated learning
if __name__ == "__main__":

    client_var = 20 #make it global later
    num_rounds = 10

    model = LogisticRegression()
    utils.set_initial_params(model, n_classes=2, n_features=10) # CHANGE based on dataset
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=client_var,
        min_fit_clients = client_var,
        min_evaluate_clients = client_var,
        fit_metrics_aggregation_fn=utils.weighted_average,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
    )

    fedmed_strategy = fl.server.strategy.FedMedian(
        min_available_clients=client_var,
        min_fit_clients = client_var,
        min_evaluate_clients = client_var,
        fit_metrics_aggregation_fn=utils.weighted_average,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
    )

    # fedavgM_strategy = fl.server.strategy.FedAvgM(
    #     min_available_clients=client_var,
    #     min_fit_clients = client_var,
    #     min_evaluate_clients = client_var,
    #     fit_metrics_aggregation_fn=utils.weighted_average,
    #     evaluate_metrics_aggregation_fn=utils.weighted_average,
    #     server_momentum = 0.9 #
    #     #server_learning_rate = 1 #default
    # )
    


    # prox_strategy = fl.server.strategy.FedProx( 
    #     fit_metrics_aggregation_fn=utils.weighted_average,
    #     evaluate_metrics_aggregation_fn=utils.weighted_average,
    #     #fraction_fit= 0.00001, # because we want the number of clients to sample on each round to be solely defined by min_fit_clients
    #     #fraction_evaluate= 0.00001,
    #     min_fit_clients= client_var,
    #     min_evaluate_clients= client_var,
    #     min_available_clients=client_var,
    #     proximal_mu =1000.0 #float!
    # )


    # opt_strategy = fl.server.strategy.FedOpt( 
    #     fit_metrics_aggregation_fn=utils.weighted_average,
    #     evaluate_metrics_aggregation_fn=utils.weighted_average,
    #     min_fit_clients= 5,
    #     min_evaluate_clients= 5,
    #     min_available_clients=5,
    #     initial_parameters = utils.get_model_parameters_tf,
    #  )


    

    

    # # Wrap the strategy with the DifferentialPrivacyServerSideFixedClipping wrapper
    # dp_strategy = DifferentialPrivacyServerSideFixedClipping(
    #     strategy,
    #     noise_multiplier=0.5,
    #     clipping_norm=5,
    #     num_sampled_clients=2,
    # )
    

    dp_strategy = DifferentialPrivacyServerSideAdaptiveClipping(
        strategy,
        noise_multiplier=0.1,
        num_sampled_clients=client_var,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy, #>>>>>>>>>>>>CHANGE STRATEGY<<<<<<<<<<<<<<
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )



    
# wipe metric file after every run!!!
# utils.plot_test_accuracy()
# utils.delete_json_file()
