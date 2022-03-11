def test(start_date,end_date,ticker_list,time_interval,technical_indicator_list,
    env,model_name,if_vix=True,**kwargs):

    from drltrading.stablebaselines3.models import DRLAgent as DRLAgent_sb3
    from drltrading.meta.data_processor import DataProcessor

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
        "if_train": False,
    }
    env_instance = env(config=env_config)

    net_dimension = kwargs.get("net_dimension", 2 ** 7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))


    episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
        model_name=model_name, environment=env_instance, cwd=cwd
    )

    return episode_total_assets