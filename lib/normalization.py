import numpy as np
import torch


def _ndim(data):
    return data.dim() if isinstance(data, torch.Tensor) else np.ndim(data)


def _shape(data):
    return tuple(data.shape)


def _expand_last_dim_for_window_scaler(data, ref):
    if _ndim(data) == 3 and _ndim(ref) == 4:
        data_shape = _shape(data)
        ref_shape = _shape(ref)
        if ref_shape[-1] == 1 and ref_shape[1] == data_shape[1]:
            if isinstance(data, torch.Tensor):
                return data.unsqueeze(-1), True
            return np.expand_dims(data, axis=-1), True
    return data, False


def _restore_last_dim(data, squeezed):
    if not squeezed:
        return data
    return data.squeeze(-1) if isinstance(data, torch.Tensor) else np.squeeze(data, axis=-1)


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def transform(self, data):
        data, squeezed = _expand_last_dim_for_window_scaler(data, self._mean)
        out = (data - self._mean) / self._std
        return _restore_last_dim(out, squeezed)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self._mean) == np.ndarray:
            self._std = torch.from_numpy(self._std).to(data.device).type(data.dtype)
            self._mean = torch.from_numpy(self._mean).to(data.device).type(data.dtype)
        data, squeezed = _expand_last_dim_for_window_scaler(data, self._mean)
        out = (data * self._std) + self._mean
        return _restore_last_dim(out, squeezed)

    @property
    def metrics_coef(self):
        return self._std


class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self._min = min
        self._max = max

    def transform(self, data):
        data, squeezed = _expand_last_dim_for_window_scaler(data, self._min)
        out = (data - self._min) / (self._max - self._min)
        return _restore_last_dim(out, squeezed)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self._min) == np.ndarray:
            self._min = torch.from_numpy(self._min).to(data.device).type(data.dtype)
            self._max = torch.from_numpy(self._max).to(data.device).type(data.dtype)
        data, squeezed = _expand_last_dim_for_window_scaler(data, self._min)
        out = data * (self._max - self._min) + self._min
        return _restore_last_dim(out, squeezed)

    @property
    def metrics_coef(self):
        return self._max - self._min


class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self._min = min
        self._max = max

    def transform(self, data):
        data, squeezed = _expand_last_dim_for_window_scaler(data, self._min)
        out = ((data - self._min) / (self._max - self._min)) * 2. - 1.
        return _restore_last_dim(out, squeezed)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self._min) == np.ndarray:
            self._min = torch.from_numpy(self._min).to(data.device).type(data.dtype)
            self._max = torch.from_numpy(self._max).to(data.device).type(data.dtype)
        data, squeezed = _expand_last_dim_for_window_scaler(data, self._min)
        out = ((data + 1.) / 2.) * (self._max - self._min) + self._min
        return _restore_last_dim(out, squeezed)

    @property
    def metrics_coef(self):
        return (self._max - self._min) / 2
