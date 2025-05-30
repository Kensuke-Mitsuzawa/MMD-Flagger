from hallucination_mt.module_flagging.module_classify_trajectory.module_classify_rule_base import classify_function_shape

import numpy as np
import math



def test_classify_function_shape():
    # Example Data
    random_seed = 42
    gen_random = np.random.default_rng(random_seed)

    sin_from = (3 * math.pi / 2)
    sin_to = 2 * math.pi
    sin_sequence_x = np.arange(sin_from, sin_to, (sin_to-sin_from) / 100)
    _sin_sequence_y = np.sin(sin_sequence_x) + np.abs(np.min(np.sin(sin_sequence_x))) * 2

    random_factors = gen_random.normal(5, 0.1, size=100)
    sin_sequence_y = _sin_sequence_y + random_factors
    sin_sequence_y[0] = 1.0

    filter_options = ['rolling_mean', 'savgol_filter', 'no_filter']
    for _filter in filter_options:
        name_shape = classify_function_shape(sin_sequence_x, sin_sequence_y, type_filter=_filter)
        assert name_shape == "monotonic-increasing"
    # end if

    x2 = np.linspace(0, 10, 100)
    y2 = np.concatenate([np.linspace(10, 0, 50), np.linspace(0, 10, 50)])

    for _filter in filter_options:
        name_shape = classify_function_shape(x2, y2)
        assert name_shape == "saddle-point"
    # end if
# end def