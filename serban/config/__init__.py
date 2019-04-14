def config_to_state(config_class):
    """
    Convert a config_class to a state dict.

    :param config_class:
    :return:
    """
    return {
        key: getattr(config_class, key)
        for key in filter(lambda name: not name.startswith('_'), dir(config_class))
    }
