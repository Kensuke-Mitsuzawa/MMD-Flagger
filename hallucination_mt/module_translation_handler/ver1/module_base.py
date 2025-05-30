import abc


class BaseTranslationModelHandler(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def sample_multiple_times(self):
        raise NotImplementedError()
