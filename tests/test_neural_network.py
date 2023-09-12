
import sys
sys.path.append("./")

import pytest

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.neural_network import Artificial_Neural_Network




def test_Artificial_Neural_Network():
    """
    This function tests the class "Artificial_Neural_Network"
    """

    ##Valid parameters
    NN = Artificial_Neural_Network(16, 4, 1)
    assert isinstance(NN.input_size, int)
    assert isinstance(NN.hidden_size, int)
    assert isinstance(NN.output_size, int)

    ##Invalid Parameters
    with pytest.raises(TypeError):
        Artificial_Neural_Network('16', 4, 1)
    with pytest.raises(TypeError):
        Artificial_Neural_Network(16, 4.5, 1)
    with pytest.raises(TypeError):
        Artificial_Neural_Network(16, 4, [1])

    
    ##Valid Fit
    Y, w, c = Artificial_Neural_Network(4, 2, 1).fit(
        np.array(
            [[0.1,0.2,0.2,0.1],
             [0.4,0.3,0.2,0.6],
             [0.4,0.3,0.2,0.6],
             [0.1,0.2,0.2,0.1]]
        ),
        np.array(
            [[0],
             [1],
             [1],
             [0]]
        ),
        alpha=0.01,
        iterations=2,
        MSE_stop=0.5
    )
    assert isinstance(Y, np.ndarray)
    assert isinstance(w, np.ndarray)
    assert isinstance(c, np.ndarray)

    ##Invalid Fit
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).fit(
            [[0.1,0.2,0.2,0.1],
            [0.4,0.3,0.2,0.6],
            [0.4,0.3,0.2,0.6],
            [0.1,0.2,0.2,0.1]],
            np.array(
                [[0],
                [1],
                [1],
                [0]]
            ),
            alpha=0.01,
            iterations=2,
            MSE_stop=0.5
        )
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).fit(
            np.array(
                [[0.1,0.2,0.2,0.1],
                [0.4,0.3,0.2,0.6],
                [0.4,0.3,0.2,0.6],
                [0.1,0.2,0.2,0.1]]
            ),
            pd.DataFrame([[0],[1],[1],[0]]),
            alpha=0.01,
            iterations=2,
            MSE_stop=0.5
        )

    
    ##Valid Pred
    P = Artificial_Neural_Network(4, 2, 1).predict(
        np.array(
            [[0.1,0.2,0.2,0.1],
             [0.4,0.3,0.2,0.6]]
        ),
        w=w,
        c=c
    )
    assert isinstance(P, np.ndarray)

    ##Invalid Pred
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).predict(
            [[0.1,0.2,0.2,0.1],
            [0.4,0.3,0.2,0.6]],
            w=w,
            c=c
        )
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).predict(
            np.array(
                [[0.1,0.2,0.2,0.1],
                [0.4,0.3,0.2,0.6]]
            ),
            w=list(w),
            c=c
        )
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).predict(
            np.array(
                [[0.1,0.2,0.2,0.1],
                [0.4,0.3,0.2,0.6]]
            ),
            w=w,
            c=0.8264
        )


    ##Valid Table
    table = Artificial_Neural_Network(4, 2, 1).probas(
        np.array([[1],[0],[0],[1],[0],[0],[0],[1],[0],[1]]),
        np.array([[0.38],[0.21],[0],[1],[0.08],[0.74],[0.12],[0.94],[0.2],[0.96]]),
        np.array(['class_1', 'class_2'])
    )
    assert isinstance(table, pd.core.frame.DataFrame)

    ##Invalid Table
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).probas(
            [[1],[0],[0],[1],[0],[0],[0],[1],[0],[1]],
            np.array([[0.38],[0.21],[0],[1],[0.08],[0.74],[0.12],[0.94],[0.2],[0.96]]),
            np.array(['class_1', 'class_2'])
        )
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).probas(
            np.array([[1],[0],[0],[1],[0],[0],[0],[1],[0],[1]]),
            [[0.38],[0.21],[0],[1],[0.08],[0.74],[0.12],[0.94],[0.2],[0.96]],
            np.array(['class_1', 'class_2'])
        )
    with pytest.raises(TypeError):
        Artificial_Neural_Network(4, 2, 1).probas(
            np.array([[1],[0],[0],[1],[0],[0],[0],[1],[0],[1]]),
            np.array([[0.38],[0.21],[0],[1],[0.08],[0.74],[0.12],[0.94],[0.2],[0.96]]),
            pd.DataFrame(np.array(['class_1', 'class_2']))
        )
