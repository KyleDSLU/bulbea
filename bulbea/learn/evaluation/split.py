# imports - compatibility packages
from __future__ import absolute_import

# imports - third-party packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# module imports
from bulbea._util import (
    _check_type,
    _check_int,
    _check_real,
    _check_iterable,
    _check_pandas_dataframe,
    _check_sequence,
    _validate_in_range
)

import bulbea as bb

def split(share, train = 0.60):
    '''
    :param attrs: `str` or `list` of attribute names of a share, defaults to *Close* attribute
    :type attrs: :obj: `str`, :obj:`list`
    '''
    _check_type(share, type_ = bb.Share, raise_err = True, expected_type_name = 'bulbea.Share')
    _check_real(train, raise_err = True)

    _validate_in_range(train, 0, 1, raise_err = True)

    if share.normalized:
        splits = share.norm_splits
    else:
        splits = share.splits

    size   = len(splits)
    split  = int(np.rint(train * size))

    train  = splits[:split,:]
    test   = splits[split:,:]

    #Xtrain, Xtest = train[:,:-1], test[:,:-1]
    Xtrain, Xtest = share.return_xcols(train), share.return_xcols(test)

    #ytrain, ytest = train[:,-1,share._splits_ycolumns], test[:,-1,share._splits_ycolumns]
    ytrain, ytest = share.return_ycols(train), share.return_ycols(test)
    share.save_split_index(split)

    return (Xtrain, Xtest, ytrain, ytest)
