from fully_connected import FullyConnectedModel


def  load_model(model_config):
    """
    Minimal wrapper to return the model. Allows for other model types in the future.
    """
    return FullyConnectedModel(model_config)
