import torch


class __Backend:
    """
    Global backend. Set global dt, device.

    DEFAULTS
    --------
    dt        = 0.005 [ms]
    """

    __defaults__ = {
        "dt": 0.005,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    _instance = None  # Keep instance reference

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        for k, v in self.__defaults__.items():
            setattr(self, f"_{k}", v)

    @property
    def dt(self):
        """Global simulation timestep."""
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set dt globally."""
        self._dt = value

    @property
    def device(self):
        """Global device."""
        return self._device

    @device.setter
    def device(self, value):
        """Set device globally."""
        self._device = value


Backend = __Backend()
