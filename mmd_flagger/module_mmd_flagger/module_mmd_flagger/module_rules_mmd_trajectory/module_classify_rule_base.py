import typing as ty
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


RETURN_VALUES = ('monotonic-increasing', 'saddle-point')
POSSIBLE_FILTERS = ('rolling_mean', 'savgol_filter', 'no_filter')


def __rule_v1(x: np.ndarray, 
              y_smooth: np.ndarray, 
              threshold: float = 0.01,
              tau_zero: ty.Optional[float] = None) -> str:
    """The rule consists of two parts,

    rule-1: the minimum y-value should be in the middle region of the x-axis.
    
    The return is 'saddle-point' when rule-1 is satisfied.    
    """
    # Find the global minimum
    min_index = np.argmin(y_smooth)
    x_min = x[min_index]
    
    # Define the middle region
    x_range = x.max() - x.min()
    lower_bound = x.min() + threshold * x_range
    upper_bound = x.max() - threshold * x_range

    if tau_zero is not None:
        # Classify based on the location of the global minimum
        if tau_zero <= x_min < upper_bound:
            return 'saddle-point'
        else:
            return 'monotonic-increasing'
    else:
        # Classify based on the location of the global minimum
        if lower_bound < x_min < upper_bound:
            return 'saddle-point'
        else:
            return 'monotonic-increasing'


def __rule_v2(x: np.ndarray, 
              y_smooth: np.ndarray, 
              spacer: float, 
              thereshold_ratio_diff: float,
              tau_zero: ty.Optional[float] = None) -> str:
    """The rule consists of two parts,
    rule-1: the minimum y-value should be in the middle region of the x-axis.
    rule-2: a ratio (the-minimum-y-value / the-first-y-value) should be greater than a threshold.

    The return is 'saddle-point' when rule-1 and rule-2 are satisfied.    
    """
    # Find the global minimum
    min_index = np.argmin(y_smooth)
    x_min = x[min_index]
    
    # Define the middle region
    x_range = x.max() - x.min()
    # lower_bound = x.min() + threshold * x_range
    # upper_bound = x.max() - threshold * x_range
    lower_bound = x.min() + spacer
    upper_bound = x.max() - spacer
    
    # check the diff values from the initial.
    y_at_initial_x = y_smooth[0]
    y_at_smallest = y_smooth[min_index]
    # If the monotonically increasing, the ratio should be at least > 0.9
    ratio_diff = y_at_smallest / y_at_initial_x

    # Classify based on the location of the global minimum
    if tau_zero is not None:
        if tau_zero <= x_min < upper_bound:
            if ratio_diff < thereshold_ratio_diff:
                return 'saddle-point'
            else:
                return 'monotonic-increasing'
        else:
            return 'monotonic-increasing'
    else:
        if lower_bound < x_min < upper_bound:
            if ratio_diff < thereshold_ratio_diff:
                return 'saddle-point'
            else:
                return 'monotonic-increasing'
        else:
            return 'monotonic-increasing'


def apply_filter(x: np.ndarray, 
                 y: np.ndarray,
                 type_filter: str,
                 window_length: ty.Optional[int],
                 polyorder: int = 2,
                 ) -> ty.Tuple[np.ndarray, np.ndarray]:
    if type_filter == "rolling_mean":
        assert window_length is not None, "Window length must be specified for filtering."
        __y_smooth = pd.Series(y).rolling(window=window_length, center=True).mean()
        # getting non-nan index
        __mask_nan = __y_smooth.isna()
        y_smooth = (__y_smooth[~__mask_nan]).to_numpy()
        # I replace the x
        x = x[~__mask_nan]
    elif type_filter == "savgol_filter":
        assert window_length is not None, "Window length must be specified for filtering."
        # Apply Savitzky-Golay filter for smoothing
        y_smooth = savgol_filter(y, window_length=min(window_length, len(y)), polyorder=polyorder)
    elif type_filter == "no_filter":
        y_smooth = y
    else:
        raise ValueError(f"Unknown filter type: {type_filter}")
    # end if

    return x, y_smooth


def classify_function_shape(x: np.ndarray, 
                            y: np.ndarray, 
                            window_length: ty.Optional[int] = 5, 
                            polyorder: int = 2, 
                            threshold: float = 0.01, 
                            spacer: float = 0.1, 
                            thereshold_ratio_diff: float = 0.9,
                            type_filter: str = "rolling_mean",
                            rule_version: str = 'v1',
                            tau_zero: ty.Optional[float] = None) -> str:
    """
    Classifies the function shape as either 'monotonic increasing' or 'saddle-point'.
    
    Parameters:
    - x: numpy array of x values (assumed to be sorted in increasing order).
    - y: numpy array of y values.
    - window_length: Window size for Savitzky-Golay filter (must be odd and <= len(y)).
    - polyorder: Polynomial order for Savitzky-Golay filter.
    - threshold: Fraction of x-range to define 'middle' for saddle-point detection.
    
    Returns:
    - str: 'monotonic increasing' or 'saddle-point'.
    """
    assert type_filter in POSSIBLE_FILTERS, f"Unknown filter type: {type_filter}"
    assert len(x) == len(y), "x and y must have the same length."
    if len(y) == 0:
        raise ValueError("y (mmd sequence) is empty. Cannot classify function shape.")
    
    if type_filter == "no_filter":
        pass
    else:
        assert window_length is not None, "Window length must be specified for filtering."
        assert len(x) > window_length, "Window length must be less than the number of data points."
    # end if

    # if type_filter == "rolling_mean":
    #     __y_smooth = pd.Series(y).rolling(window=window_length, center=True).mean()
    #     # getting non-nan index
    #     __mask_nan = __y_smooth.isna()
    #     y_smooth = (__y_smooth[~__mask_nan]).to_numpy()
    #     # I replace the x
    #     x = x[~__mask_nan]
    # elif type_filter == "savgol_filter":
    #     # Apply Savitzky-Golay filter for smoothing
    #     y_smooth = savgol_filter(y, window_length=min(window_length, len(y)), polyorder=polyorder)
    # elif type_filter == "no_filter":
    #     y_smooth = y
    # else:
    #     raise ValueError(f"Unknown filter type: {type_filter}")
    # # end if

    x, y_smooth = apply_filter(x=x, 
                               y=y, 
                               type_filter=type_filter, 
                               window_length=window_length, 
                               polyorder=polyorder)

    if rule_version == 'v1':
        shape = __rule_v1(x, y_smooth, threshold, tau_zero=tau_zero)
    elif rule_version == 'v2':
        shape = __rule_v2(x, y_smooth, spacer, thereshold_ratio_diff, tau_zero=tau_zero)
    else:
        raise ValueError(f"Unknown rule version: {rule_version}")
    # end if

    return shape
