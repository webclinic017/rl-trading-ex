from datetime import datetime
from data import DataProcessor
from ddpg import DDPG
from ddqn import DDQN
from env import StockTradingEnv, PortfolioAllocationEnv
import matplotlib.pyplot as plt
# import pandas as pd
from ray import tune
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.schedulers.async_hyperband import ASHAScheduler
import wandb
import warnings
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
plt.ioff()

def load_data(parameters):
    data_path = parameters["config"]["data_path"]

    train_path = data_path + "/train.csv"
    test_path = data_path + "/test.csv"

    data = DataProcessor(train_path, test_path)
    data.load_data()
    data.calc_ind()
    data.scale_ind()
    df_train, df_test = data.get_df()

    return df_train, df_test


def make_envs(df_train, df_test, parameters, env_type="StockTrading"):
    
    env_config = parameters["env"][env_type]

    if env_type == "StockTrading":
        env_train = StockTradingEnv(df=df_train, env_config=env_config)
        env_test = StockTradingEnv(df=df_test, env_config=env_config)
    elif env_type == "PortfolioAllocation":
        env_train = PortfolioAllocationEnv(df=df_train, env_config=env_config)
        env_test = PortfolioAllocationEnv(df=df_test, env_config=env_config)
    else:
        raise ValueError("Unknown environment type.")

    return env_train, env_test


def parse_parameters(parameters, model_name):
    model_config = parameters["models"][model_name]["train"]
    tune_config = parameters["models"][model_name]["tune"]

    

    #     # define search space here
    #     "a": tune.choice([1, 2, 3]),
    #     "b": tune.choice([4, 5, 6]),
    #     # wandb configuration
    #     "wandb": {
    #         "project": "Optimization_Project",
    #         "api_key_file": "/path/to/file",
    #     },
    # }

    # TODO: Check for previous best_config and update if only running train

    return model_config, tune_config


def plot(env):  # TODO: Make this work for tune as well
    x = env.df.index
    sharpe_memory = env.sharpe_memory
    trades = env.trades
    portfolio = env.portfolio_value
    stock_price = env.stock_price_memory

    figure, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].plot(x, stock_price)
    axes[0, 1].plot(x, portfolio)
    axes[1, 0].plot(x, trades)
    axes[1, 1].plot(x, sharpe_memory)
    figure.tight_layout()

    return figure


def train(model_name="DDPG", metric_name="Sharpe", parameters=None, log=False):
    if parameters is None:
        raise ValueError("No parameters passed.")

    df_train, df_test = get_data(parameters)
    env_train, env_test = make_envs(df_train, df_test, parameters)
    model_config, tune_config = parse_parameters(parameters, model_name)

    if model_name == "DDPG":
            model = DDPG(
                model_config["state_space"],
                model_config["buffer_size"],
                model_config["hidden_p"],
                model_config["hidden_q"],
                model_config["lr_p"],
                model_config["lr_q"],
                model_config["gamma"],
                model_config["k"],
                model_config["tau"],
                model_config["var_k"],
                model_config["explore_pc"],
                model_config["iters"],
                model_config["train_cycle"],
                model_config["hmax"],
            )
    elif model_name == "DDQN":
        model = DDQN()  # TODO: Add parameters here
    else:
        raise ValueError("Unknown model name.")

    wandb.log({"architecture": model_name})



    loss = model.train(env_train)
    _, sharpe, total_reward = model.evaluate(env_test)

    if metric_name == "Sharpe":
        metric = sharpe
    elif metric_name == "Total_Reward":
        metric = total_reward
    else:
        raise ValueError("Unknown metric name.")

    if log:
        wandb.log({"architecture": model_name})
        for key in model_config:
            wandb.log({key: model_config[key]})
        
        wandb.log({"metric": metric_name})

    figure = plot(env_test)

    return figure, metric


def tune(model_name="DDPG", metric_name="Sharpe", parameters=None):
    if parameters is None:
        raise ValueError("No parameters passed.")

    df_train, df_test = get_data(parameters)
    env_train, env_test = make_envs(df_train, df_test, parameters)
    model_config, tune_config = parse_parameters(parameters, model_name)

    @wandb_mixin
    def train_fn(tune_config):
        for k in tune_config:
            if k in model_config:
                model_config[k] = tune_config[
                    k
                ]  # Run model with tune parameters if specified

        if model_name == "DDPG":
            model = DDPG( # TODO: Move this to model init
                model_config["state_space"],
                model_config["buffer_size"],
                model_config["hidden_p"],
                model_config["hidden_q"],
                model_config["lr_p"],
                model_config["lr_q"],
                model_config["gamma"],
                model_config["k"],
                model_config["tau"],
                model_config["var_k"],
                model_config["explore_pc"],
                model_config["iters"],
                model_config["train_cycle"],
                model_config["hmax"],
            )
        elif model_name == "DDQN":
            model = DDQN()  # TODO: Add parameters here
        else:
            raise ValueError("Unknown model name.")

        wandb.log({"architecture": model_name})

        for key in model_config:
            wandb.log({key: model_config[key]})

        loss = model.train(env_train)

        for key in loss:
            wandb.log({key: loss[key]})

        _, sharpe, total_reward = model.evaluate(env_test)

        if metric_name == "Sharpe":
            metric = sharpe
        elif metric_name == "Total_Reward":
            metric = total_reward
        else:
            raise ValueError("Unknown metric name.")

        wandb.log({metric_name: metric})

        yield {metric_name: metric}

    asha_scheduler = ASHAScheduler(metric=metric_name)

    analysis = tune.run(
        train_fn,
        config=tune_config,
        scheduler=asha_scheduler,
    )

    df_analysis = analysis.results_df

    best_config = analysis.get_best_config(metric=metric_name, mode="max")

    return df_analysis, best_config


if __name__ == "__main__":

    with open("_parameters.yaml") as f:
        parameters = yaml.safe_load(f)

    metric_name = parameters["config"]["metric"]

    model_name = parameters["config"]["model_name"]

    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    if parameters["config"]["run_type"] == "tune":
        if model_name == "all":
            for key in parameters["models"]:
                df_analysis, best_config = tune(
                    model_name=key, metric_name=metric_name, parameters=parameters
                )
                df_analysis.to_csv(
                    parameters["results_path"]
                    + "/df_analysis_"
                    + key
                    + "_"
                    + now
                    + ".csv"
                )
                with open(
                    parameters["results_path"]
                    + "/best_config_"
                    + key
                    + "_"
                    + now
                    + ".yaml",
                    "w",
                ) as f:
                    yaml.safe_dump(best_config, f)
        else:
            df_analysis, best_config = tune(
                model_name=model_name, metric_name=metric_name, parameters=parameters
            )
            df_analysis.to_csv(
                parameters["results_path"]
                + "/df_analysis_"
                + model_name
                + "_"
                + now
                + ".csv"
            )
            with open(
                parameters["results_path"]
                + "/best_config_"
                + model_name
                + "_"
                + now
                + ".yaml",
                "w",
            ) as f:
                yaml.safe_dump(best_config, f)

    elif parameters["config"]["run_type"] == "train":



    else:
        raise ValueError("Incorrect run type selected.")
