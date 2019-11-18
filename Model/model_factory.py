import Model


def ModelFactory(**kwargs):
    """
    Used to load different models
    """
    available_models = []
    model = kwargs.get("model_name")
    if not hasattr(Model, model):
        raise ValueError("Please select a proper model. \n The selected model '{}' does not exists".format(model))

    return getattr(Model, model)(kwargs)

