from drltrading.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from drltrading.meta.data_processor import DataProcessor


def train(
        start_date,
        end_date,
        ticker_list,
        time_interval,
        technical_indicator_list,
        env,
        model_name,
        if_vix=True,
        **kwargs
):
    # fetch data
    DP = DataProcessor( **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    total_timesteps = kwargs.get("total_timesteps", 1e6)
    agent_params = kwargs.get("agent_params")

    agent = DRLAgent_sb3(env=env_instance)

    model = agent.get_model(model_name, model_kwargs=agent_params)
    trained_model = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=total_timesteps
    )
    print("Training finished!")
    trained_model.save(cwd)
    print("Trained model saved in " + str(cwd))