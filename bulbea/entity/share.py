# imports - compatibility packages
from __future__ import absolute_import
# imports - standard packages
import os
import warnings
# imports - third-party packages
import numpy as np
import matplotlib.pyplot as pplt
import pandas as pd
import quandl

# module imports
from sklearn.preprocessing import MinMaxScaler
from bulbea.config.app import AppConfig
from bulbea.entity import Entity
from bulbea._util import (
    _check_type,
    _check_str,
    _check_int,
    _check_real,
    _check_pandas_series,
    _check_pandas_dataframe,
    _check_pandas_timestamp,
    _check_iterable,
    _check_environment_variable_set,
    _validate_in_range,
    _validate_date,
    _assign_if_none,
    _get_type_name,
    _get_datetime_str,
    _raise_type_error,
    _is_sequence_all
)
from bulbea._util.const import (
    ABSURL_QUANDL,
    QUANDL_MAX_DAILY_CALLS,
    SHARE_ACCEPTED_SAVE_FORMATS
)
from bulbea._util.color import Color
import bulbea as bb

pplt.style.use(AppConfig.PLOT_STYLE)

class IndexObject:
    def __init__(self, columns):
        self.columns = {column: i for i, column in enumerate(columns)}

def _sync_pandas_dataframe_ix(df1, df2):
    _check_pandas_dataframe(df1, raise_err = True)
    _check_pandas_dataframe(df2, raise_err = True)
    ix1 = df1.index
    ix2 = df2.index
    if ix1.equals(ix2):
        return (df1, df2)
    diff = ix1.difference(ix2)
    diff = diff.append(ix2.difference(ix1))
    for ix in diff:
        try:
            df1 = df1.drop(ix)
        except KeyError:
            pass
        try:
            df2 = df2.drop(ix)
        except KeyError:
            pass
    return (df1, df2)

def _get_cummulative_return(data, base):
    #cumret  = (data / data[0]) - 1
    if base == 0:
        cumret = data*0
    else:
        cumret  = (data / base) - 1
    return cumret

def _get_period_slope(data, period):
    _check_pandas_dataframe(data, raise_err = True)
    shift = data.shift(period)
    ret = (data - shift)/data
    return ret

def _get_encoded_days(data):
    _check_pandas_dataframe(data, raise_err = True)
    index = data.index
    days = {"Monday": 0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}
    encoded_days = [[1 if indice.weekday()==day_val else 0 for day_val in days.values()] for indice in index]
    encoded_days = pd.DataFrame(encoded_days, columns=days.keys(), index=index)
    return encoded_days

def _get_bollinger_bands_columns(data):
    _check_pandas_dataframe(data, raise_err = True)

    columns   = list(data.columns)
    ncols     =  len(columns)

    if ncols != 3:
        raise ValueError('Expected a pandas.DataFrame with exactly 3 columns, got {ncols} instead.'.format(
            ncols = ncols
        ))

    if not _is_sequence_all(columns):
        raise ValueError('Ambiguous column names: {columns}'.format(
            columns = columns
        ))

    attr       = columns[0]
    prefixes   = ['Lower', 'Mean', 'Upper']
    columns    = ['{prefix}_{attr}'.format(
        prefix = prefix,
        attr   = attr
    ) for prefix in prefixes]

    return columns

def _get_aroon_oscillator(data, period=152):
    _check_pandas_dataframe(data, raise_err = True)
    high_roll = data['High'].rolling(window=period)
    low_roll = data['Low'].rolling(window=period)
    aroon = pd.DataFrame(columns=["Up","Down"], index=data.index)
    aroon["Up"] = (period-high_roll.apply(np.argmax, raw=True))/period * 100
    aroon["Down"] = (period-low_roll.apply(np.argmin, raw=True))/period * 100
    return aroon["Up"] - aroon["Down"]

def _get_roc(data):
    _check_pandas_dataframe(data, raise_err = True)
    time = ['t+0', 't-1', 't-2']
    col = ["ROC {time} ({attr})".format(time = t, attr = attribute) for attribute in data.columns for t in time]

    ix = data.index
    roc = pd.DataFrame(np.nan, columns=col, index=ix)
    pn = data.shift(1)
    roc[col[0]] = (data/pn - 1) * 100
    for i in range(1, len(col)):
        roc[col[i]] = roc[col[i-1]].shift(1)
    test = pd.concat([data["Close"], roc[col]], axis=1)
    return roc

def _get_cmo(data, period):
    _check_pandas_dataframe(data, raise_err = True)
    _check_int(period, raise_err = True)
    cmo = pd.DataFrame(columns=['CMO'], index=data.index)
    s = pd.DataFrame(columns=['PosSum','NegSum'], index=data.index)
    s['PosSum'] = np.where(data > data.shift(1), data + data.shift(1), 0)
    s['NegSum'] = np.where(data <= data.shift(1), data + data.shift(1), 0)
    pos_roll = s['PosSum'].rolling(window = period)
    neg_roll = s['NegSum'].rolling(window = period)
    cmo = (pos_roll.sum() - neg_roll.sum())/(pos_roll.sum() + neg_roll.sum()) * 100
    return cmo

def _get_ichimoku(data):
    _check_pandas_dataframe(data, raise_err = True)
    ix = data.index
    ichimoku = pd.DataFrame(columns=["TL","SL","LS1","LS2","Cloud"],  
                                    index=ix) 

    roll_9 = data.rolling(window = 9)
    roll_26 = data.rolling(window = 26)
    roll_52 = data.rolling(window = 52)
    ichimoku['TL'] = (roll_9['High'].max() + roll_9['Low'].min())/2
    ichimoku['SL'] = (roll_26['High'].max() + roll_26['Low'].min())/2
    ichimoku['LS1'] = ichimoku['TL'] + ichimoku['SL']
    ichimoku['LS2'] = (roll_52['High'].max() + roll_52['Low'].min())/2
    ichimoku['Cloud'] = (ichimoku['LS1'] - ichimoku['LS2']).rolling(window=26).sum()
    return ichimoku

def _get_ranged_high_m_low(data):
    _check_pandas_dataframe(data, raise_err = True)
    
    r = 3
    ranged_high = data['High'][::-1].rolling(window=r).max()[::-1]
    high_low = pd.DataFrame(columns=["HighR3-Low"], index=data.index)
    high_low["HighR3-Low"] = (ranged_high - data["Low"])/data["Low"]

    return high_low

def _get_time_close(data, tt_hour, tt_min, ticker)
    _check_int(tt, raise_err = True)
    _check_pandas_series(data, raise_err = True)
    envvar = AppConfig.ENVIRONMENT_VARIABLE['local_ohlc_storage']
    if not _check_environment_variable_set(envvar):
        message = Color.warn("Local ohlc storage not defined.")
        warnings.warn(message)
        return None
    else:
        # ASSUMING 1d OHLC STORAGE -- TO BE FIXED
        local_storage_path = os.path.join(os.getenv(envvar), '5m/')
    local_storage_path = os.path.join(local_storage_path, ticker + '.h5')
    if os.path.exists(local_storage_path):
        hdf = pd.HDFStore(local_storage_path)
        df = hdf.get(ticker)
        col = "{h}:{m}_Close".format(h=tt_hour, m=tt_min)
        tt = pd.DataFrame(columns=[col], index=df.index)
        tt[col] = df.at_time('{h}:{m}'.format(h=tt_hour, m=tt_min))
        tt.index = tt.index.tz_localize(None).normalize()
        return tt
    else:
        raise ValueError("Local OHLC Storage Not Found")

def _get_time_high(data, tt_hour, tt_min, ticker)
    _check_int(tt, raise_err = True)
    _check_pandas_series(data, raise_err = True)
    envvar = AppConfig.ENVIRONMENT_VARIABLE['local_ohlc_storage']
    if not _check_environment_variable_set(envvar):
        message = Color.warn("Local ohlc storage not defined.")
        warnings.warn(message)
        return None
    else:
        # ASSUMING 1d OHLC STORAGE -- TO BE FIXED
        local_storage_path = os.path.join(os.getenv(envvar), '5m/')
    local_storage_path = os.path.join(local_storage_path, ticker + '.h5')
    if os.path.exists(local_storage_path):
        hdf = pd.HDFStore(local_storage_path)
        df = hdf.get(ticker)
        col = "{h}:{m}_High".format(h=tt_hour, m=tt_min)
        tt = pd.DataFrame(columns=[col], index=df.index.normalize().tz_localize(None).drop_duplicates())
        tt[col] = [df.between_time("9:30", "{h}:{m}".format(h=tt_hour, m=tt_min)).loc["{y}-{m}-{d}".format(y=day.year, m=day.month, d=day.day)]["High"].max() for day in tt.index]
        return tt
    else:
        raise ValueError("Local OHLC Storage Not Found")

def _get_time_low(data, tt_hour, tt_min, ticker)
    _check_int(tt, raise_err = True)
    _check_pandas_series(data, raise_err = True)
    envvar = AppConfig.ENVIRONMENT_VARIABLE['local_ohlc_storage']
    if not _check_environment_variable_set(envvar):
        message = Color.warn("Local ohlc storage not defined.")
        warnings.warn(message)
        return None
    else:
        # ASSUMING 1d OHLC STORAGE -- TO BE FIXED
        local_storage_path = os.path.join(os.getenv(envvar), '5m/')
    local_storage_path = os.path.join(local_storage_path, ticker + '.h5')
    if os.path.exists(local_storage_path):
        hdf = pd.HDFStore(local_storage_path)
        df = hdf.get(ticker)
        col = "{h}:{m}_Low".format(h=tt_hour, m=tt_min)
        tt = pd.DataFrame(columns=[col], index=df.index.normalize().tz_localize(None).drop_duplicates())
        tt[col] = [df.between_time("9:30", "{h}:{m}".format(h=tt_hour, m=tt_min)).loc["{y}-{m}-{d}".format(y=day.year, m=day.month, d=day.day)]["Low"].min() for day in tt.index]
        return tt
    else:
        raise ValueError("Local OHLC Storage Not Found")

def _get_time_volume(data, tt_hour, tt_min, ticker)
    _check_int(tt, raise_err = True)
    _check_pandas_series(data, raise_err = True)
    envvar = AppConfig.ENVIRONMENT_VARIABLE['local_ohlc_storage']
    if not _check_environment_variable_set(envvar):
        message = Color.warn("Local ohlc storage not defined.")
        warnings.warn(message)
        return None
    else:
        # ASSUMING 1d OHLC STORAGE -- TO BE FIXED
        local_storage_path = os.path.join(os.getenv(envvar), '5m/')
    local_storage_path = os.path.join(local_storage_path, ticker + '.h5')
    if os.path.exists(local_storage_path):
        hdf = pd.HDFStore(local_storage_path)
        df = hdf.get(ticker)
        col = "{h}:{m}_Volume".format(h=tt_hour, m=tt_min)
        tt = pd.DataFrame(columns=[col], index=df.index.normalize().tz_localize(None).drop_duplicates())
        tt[col] = [df.between_time("9:30", "{h}:{m}".format(h=tt_hour, m=tt_min)).loc["{y}-{m}-{d}".format(y=day.year, m=day.month, d=day.day)]["Volume"].sum() for day in tt.index]
        return tt
    else:
        raise ValueError("Local OHLC Storage Not Found")
    
def _get_close_m_52w_high_low(data):
    _check_pandas_dataframe(data, raise_err = True)

    n = int(252/52 * 52)

    high_low = pd.DataFrame(np.nan, columns=["52W_High", "52W_Low"], index = data.index)
    high_low["52W_High"] = (data['High'].rolling(window = n).max() - data['Close']) / data['Close']
    high_low["52W_Low"] = (data['Low'].rolling(window = n).min() - data['Close']) / data['Close']

    return high_low

def _get_consecutive_closings(data):
    _check_pandas_dataframe(data, raise_err = True)
    closings = pd.DataFrame(columns=["CC"], index=data.index)
    cc = 0
    close_diff = data['Close'] - data['Close'].shift(1)
    for i in range(1, len(data.index)):
        if close_diff.iloc[i] > 0 and cc >= 0:
            cc += 1
        elif close_diff.iloc[i] < 0 and cc <= 0:
            cc -= 1
        else:
            cc = 0
        closings['CC'].iloc[i] = cc
    return closings

def _get_awesome(data):
    _check_pandas_dataframe(data, raise_err = True)

    columns   = list(data.columns)
    ncols     =  len(columns)

    attrs = ["High", "Low"]
    df = data[attrs]
    avg = pd.Series((df['High'] + df['Low'])/2)

    roll_34 = avg.rolling(window = 34)
    roll_5 = avg.rolling(window = 5)
    mean_34 = roll_34.mean()
    mean_5 = roll_5.mean()

    awesome = mean_5 - mean_34
    return awesome

def _get_rsi(data):
    _check_pandas_dataframe(data, raise_err = True)
    window = 14
    close = pd.DataFrame(columns=["Diff", "Gain", "Loss"], index=data.index)
    close["Diff"] = (data['Close']-data['Close'].shift(1))/data['Close']
    mask = close["Diff"] < 0
    close["Gain"] = close["Diff"].mask(mask)
    close["Loss"] = close["Diff"].mask(~mask)

    close_gain_count = close["Gain"].rolling(window=window).count().fillna(0)
    avg_gain = close["Gain"].fillna(0).rolling(window=window).sum()/close_gain_count

    close_loss_count = close["Loss"].rolling(window=window).count().fillna(0)
    avg_loss = close["Loss"].fillna(0).rolling(window=window).sum()/close_loss_count

    rsi = pd.DataFrame(columns=["RSI"], index=data.index)
    rsi["RSI"] = 1 - (1/(1+avg_gain/avg_loss))
    return rsi

def _get_bollinger_bands(data, period = 50, bandwidth = 1):
    _check_int(period,    raise_err = True)
    _check_int(bandwidth, raise_err = True)

    _check_pandas_series(data, raise_err = True)

    roll      = data.rolling(window = period)
    std, mean = roll.std(), roll.mean()

    upper     = mean + bandwidth * std
    lower     = mean - bandwidth * std

    return (lower, mean, upper)

class Share(Entity):
    '''
    A user-created :class:`Share <bulbea.Share>` object.

    :param source: *source* symbol for economic data
    :type source: :obj:`str`

    :param ticker: *ticker* symbol of a share
    :type ticker: :obj:`str`

    :param start: starting date string in the form YYYY-MM-DD for acquiring historical records, defaults to the earliest available records
    :type start: :obj:`str`

    :param end: ending date string in the form YYYY-MM-DD for acquiring historical records, defaults to the latest available records
    :type end: :obj:`str`

    :param latest: acquires the latest N records
    :type latest: :obj:`int`

    :Example:

    >>> import bulbea as bb
    >>> share = bb.Share(source = 'YAHOO', ticker = 'GOOGL')
    >>> share.data.sample(1)
                Open       High        Low  Close      Volume  Adjusted Close
    Date
    2003-05-15  18.6  18.849999  18.470001  18.73  71248800.0        1.213325
    '''
    def __init__(self, ticker, source = None, start = None, end = None, latest = None, cache = False, local_update=False):
        _check_str(ticker, raise_err = True)

        self.source    = source
        self.ticker    = ticker

        self.update(start = start, end = end, latest = latest, cache = cache, local_update = local_update)
        self.comps = {}
        self.num_comps = 0
        self._splits_base = 5

    def update(self, start = None, end = None, latest = None, cache = False, local_update = False):
        '''
        Update the share with the latest available data.

        :Example:

        >>> import bulbea as bb
        >>> share = bb.Share(source = 'YAHOO', ticker = 'AAPL')
        >>> share.update()
        '''
        if not local_update:
            _check_str(self.source, raise_err = True)
            envvar = AppConfig.ENVIRONMENT_VARIABLE['quandl_api_key']
            if not _check_environment_variable_set(envvar):
                message = Color.warn("Environment variable {envvar} for Quandl hasn't been set. A maximum of {max_calls} calls per day can be made. Visit {url} to get your API key.".format(envvar = envvar, max_calls = QUANDL_MAX_DAILY_CALLS, url = ABSURL_QUANDL))

                warnings.warn(message)
            else:
                quandl.ApiConfig.api_key = os.getenv(envvar)

            self.data    = quandl.get('{database}/{code}'.format(
                database = self.source,
                code     = self.ticker
            ))
        else:
            envvar = AppConfig.ENVIRONMENT_VARIABLE['local_ohlc_storage']
            if not _check_environment_variable_set(envvar):
                message = Color.warn("Local ohlc storage not defined.")
                warnings.warn(message)
                return None
            else:
                # ASSUMING 1d OHLC STORAGE -- TO BE FIXED
                local_storage_path = os.path.join(os.getenv(envvar), '1d/')
                local_storage_path = os.path.join(local_storage_path, self.ticker + '.h5')
                hdf = pd.HDFStore(local_storage_path)
                df = hdf.get(self.ticker)
                hdf.close()
                if start == None:
                    start = df.index[0]
                if end == None:
                    end = df.index[-1]
                self.data = df.loc[start:end]

        self.length  =  len(self.data)
        self.attrs   = list(self.data.columns)

    def attach_competitor(self, ticker, source = None, start = None, end = None, latest = None, cache = False, local_update=False):
        self.comps[ticker] = Share(ticker, source, start, end, latest, cache, local_update)
        self.data, self.comps[ticker].data = _sync_pandas_dataframe_ix(self.data, self.comps[ticker].data)
        self.num_comps += 1

    def __len__(self):
        '''
        Number of data points available for a given share.

        :Example:
        >>> import bulbea as bb
        >>> share = bb.Share(source = 'YAHOO', ticker = 'AAPL')
        >>> len(share)
        9139
        '''
        return self.length

    def close_m_high_low_52w(self):
        high_low = _get_close_m_52w_high_low(self.data)
        return high_low

    def awesome(self):
        '''
        :Example:
        '''

        awesome = _get_awesome(self.data)
        awesome = pd.DataFrame(awesome, columns=["Awesome"])
        return awesome

    def roc(self,
            attrs     = 'Close'):
        '''
        :Example:
        '''
        _check_iterable(attrs, raise_err = True)

        if _check_str(attrs):
            attrs = [attrs]

        data = self.data[attrs]
        roc  = pd.DataFrame(_get_roc(data))

        return roc

    def close_slope(self):
        return self.week_month_3month_slope('Close')

    def week_month_3month_slope(self, attr='Close'):
        data = pd.DataFrame(self.data[attr])
        columns = ['1W_', '1M_', '3M_', '1Y_']
        columns = [col+attr for col in columns]

        slope = [_get_period_slope(data, days) for days in [5, 20, 60, 252]]
        slope = pd.concat(slope, axis=1, sort=True)
        slope.columns = columns

        return slope

    def encoded_days(self):
        data = self.data[['Close']]
        return _get_encoded_days(data)

    def ichimoku(self):
        '''
        :Example:
        '''

        data = self.data[['High','Low']]
        ichimoku = pd.DataFrame(_get_ichimoku(data))

        return pd.DataFrame(ichimoku['Cloud'])

    def cmo(self, period = 50):
        '''
        :Example:
        '''

        data = self.data[['Close']]
        cmo  = pd.DataFrame(_get_cmo(data, period), columns=['CMO'])
        return cmo

    def aro(self, period = 50):
        '''
        :Example:
        '''

        data = self.data[['High','Low']]
        aro  = pd.DataFrame(_get_aroon_oscillator(data, period), columns=['ARO'])
        return aro

    def high_open_diff(self):
        col1 = 'High'
        col2 = 'Open'
        data = self.data[[col1, col2]]
        diff = pd.DataFrame((data[col1] - data[col2])/data[col2], columns=['{col1}-{col2}'.format(col1=col1, col2=col2)])
        return diff

    def ranged_high_low_diff(self):
        data = self.data[['High', 'Low']]
        diff = _get_ranged_high_m_low(data)
        return diff

    def consecutive_closings(self):
        return _get_consecutive_closings(self.data)

    def open_shift(self):
        data = self.data['Open'].shift(-1)
        open_shift = pd.DataFrame(data.values, columns=['Open+1'], index=data.index)
        return open_shift

    def low_shift(self):
        data = self.data['Low'].shift(-1)
        open_shift = pd.DataFrame(data.values, columns=['Low+1'], index=data.index)
        return open_shift

    def prediction_time_close(self):
        return _get_time_close(12, 30, self.ticker)

    def prediction_time_low(self):
        return _get_time_low(12, 30, self.ticker)

    def prediction_time_high(self):
        return _get_time_high(12, 30, self.ticker)

    def prediction_time_volume(self):
        return _get_time_volume(12, 30, self.ticker)

    def trading_time_close(self):
        return _get_time_close(15, 15, self.ticker)

    def trading_time_low(self):
        return _get_time_low(15, 15, self.ticker)

    def trading_time_high(self):
        return _get_time_high(15, 15, self.ticker)

    def trading_time_volume(self):
        return _get_time_volume(15, 15, self.ticker)

    def rsi(self):
        return _get_rsi(self.data)

    def bolban_close(self):
        return self.bollinger_bands(attrs='Close', period=25)

    def bolban_high(self):
        return self.bollinger_bands(attrs='High', period=25)

    def bolban_low(self):
        return self.bollinger_bands(attrs='Low', period=25)

    def bollinger_bands(self,
                        attrs     = 'Close',
                        period    = 50,
                        bandwidth = 1):
        '''
        Returns the Bollinger Bands (R) for each attribute.

        :param attrs: `str` or `list` of attribute name(s) of a share, defaults to *Close*
        :type attrs: :obj:`str`, :obj:`list`

        :param period: length of the window to compute moving averages, upper and lower bands
        :type period: :obj:`int`

        :param bandwidth: multiple of the standard deviation of upper and lower bands
        :type bandwidth: :obj:`int`

        :Example:

        >>> import bulbea as bb
        >>> share     = bb.Share(source = 'YAHOO', ticker = 'AAPL')
        >>> bollinger = share.bollinger_bands()
        >>> bollinger.tail()
                    Lower (Close)  Mean (Close)  Upper (Close)
        Date
        2017-03-07     815.145883    831.694803     848.243724
        2017-03-08     816.050821    832.574004     849.097187
        2017-03-09     817.067353    833.574805     850.082257
        2017-03-10     817.996674    834.604404     851.212135
        2017-03-13     819.243360    835.804605     852.365849
        '''
        _check_iterable(attrs, raise_err = True)

        if _check_str(attrs):
            attrs = [attrs]

        frames = list()

        for attr in attrs:
            data                    = self.data[attr]
            lowr, mean, upper       = _get_bollinger_bands(data, period = period, bandwidth = bandwidth)
            bollinger_bands         = pd.concat([lowr, mean, upper], axis = 1)
            bollinger_bands.columns = _get_bollinger_bands_columns(bollinger_bands)

            frames.append(bollinger_bands)

        return frames[0] if len(frames) == 1 else frames

    def build_data_prerequisites(self, prereq, comp_prereq=None):
        self.prereq = prereq
        self.comp_prereq = comp_prereq

    def build_data(self):
        try:
            _check_iterable(self.prereq, raise_err = True)
            df = self.data
            for op in self.prereq:
                df = pd.concat([df, op()], axis=1, sort=True)
                df = df.dropna()
            if self.comp_prereq:
                for op in self.comp_prereq:
                    df = pd.concat([df, op()], axis=1, sort=True)
                    df = df.dropna()
            self.features = df
        except NameError as e:
            raise Exception("Share prerequisites must be set")

    def build_splits(self,
                     Xattrs      = ['Close'],
                     yattrs      = ['Close'],
                     cum_norm_attrs  = ['Close'],
                     norm_attrs  = [],
                     window      = 0.01,
                     shift       = 1,
                     normalize   = False):

        _check_iterable(Xattrs, raise_err = True)
        _check_iterable(yattrs, raise_err = True)
        _check_iterable(cum_norm_attrs, raise_err = True)
        _check_iterable(norm_attrs, raise_err = True)
        _check_int(shift, raise_err = True)
        _check_real(window, raise_err = True)
        _validate_in_range(window, 0, 1, raise_err = True)

        try:
            df = self.features
        except AttributeError as e:
            print('Features not built yet, building now...')
            self.build_data()
            df = self.features

        data = df[Xattrs].values
        ycolumns = []

        for i, attr in enumerate(df[Xattrs].columns):
            if attr in yattrs:
                ycolumns.append(i)

        length = len(df.index)
        window = 15
        offset = shift - 1

        splits = np.array([data[i + offset: i + window] for i in range(length - window + 1)])
        index = df.index.drop(df.index[:window-1])
        splits_ix = {ix: i for i, ix in enumerate(index)}
        self.columns = IndexObject(df[Xattrs].columns)

        self.splits = np.zeros_like(splits)
        self.splits[:,:,:] = splits[:,:,:]
        self.splits_ix = splits_ix

        self._splits_Xattr = Xattrs
        self._splits_yattr = yattrs
        self._splits_ycolumns = ycolumns
        self._splits_window = window

        self.normalized = normalize

        if normalize:
            cum_normalize_col = [i for i, attr in enumerate(df[Xattrs].columns) if attr in cum_norm_attrs]
            _found = [attr for attr in df[Xattrs].columns if attr in cum_norm_attrs]
            for i, split in enumerate(splits):
                for j in range(split.shape[1]):
                    if j in cum_normalize_col:
                        splits[i,:,j] = _get_cummulative_return(split[:,j], split[-self._splits_base-1,j])
            self._splits_cumnormattr = _found

            if norm_attrs:
                normalize_col = [i for i, attr in enumerate(df[Xattrs].columns) if attr in norm_attrs]
                _found = [attr for attr in df[Xattrs].columns if attr in norm_attrs]
                for i in range(splits.shape[2]):
                    if i in normalize_col:
                        scalar = MinMaxScaler(feature_range=(-1,1))
                        splits[:,:,i] = scalar.fit_transform(splits[:,:,i])
                self._splits_normcolumns = normalize_col
                self._splits_normattr = _found

            self.norm_splits = np.zeros_like(splits)
            self.norm_splits[:,:,:] = splits[:,:,:]

    def save_split_index(self, ix):
        _check_int(ix, raise_err = True)
        self._splits_index = ix

    def save(self, format_ = 'csv', filename = None):
        '''
        :param format_: type of format to save the Share object, default 'csv'.
        :type format_: :obj:`str`
        '''
        if format_ not in SHARE_ACCEPTED_SAVE_FORMATS:
            raise ValueError('Format {format_} not accepted. Accepted formats are: {accepted_formats}'.format(
                format_          = format_,
                accepted_formats = SHARE_ACCEPTED_SAVE_FORMATS
            ))

        if filename is not None:
            _check_str(filename, raise_err = True)
        else:
            filename = _get_share_filename(self, extension = format_)

        if format_ is 'csv':
            self.data.to_csv(filename)

    def return_data(self):
        return self.data

    def return_xcols(self, data):
        _check_type(data, np.ndarray, raise_err = True, expected_type_name = 'np.array')
        size = data.shape
        if len(size) != 3:
            raise ValueError('Expected a np.array with size 3, got {l} instead.'.format(
                l = len(size)
            ))
        return data[:,:-1]

    def return_ycols(self, data):
        _check_type(data, np.ndarray, raise_err = True, expected_type_name = 'np.array')
        size = data.shape
        if len(size) != 3:
            raise ValueError('Expected a np.array with size 3, got {l} instead.'.format(
                l = len(size)
            ))
        return data[:,-1,self._splits_ycolumns]

    def return_splits(self):
        if self.normalized:
            return self.norm_splits
        else:
            return self.splits

    def return_split_from_date(self, dates):
        return np.array([self.return_splits()[self.splits_ix[date]] for date in dates])

    def split(self, train = 0.60):
        '''
        :param attrs: `str` or `list` of attribute names of a share, defaults to *Close* attribute
        :type attrs: :obj: `str`, :obj:`list`
        '''
        _check_real(train, raise_err = True)

        _validate_in_range(train, 0, 1, raise_err = True)

        if self.normalized:
            splits = self.norm_splits
        else:
            splits = self.splits

        size   = len(splits)
        split  = int(np.rint(train * size))

        train  = splits[:split,:]
        test   = splits[split:,:]

        Xtrain, Xtest = self.return_xcols(train), self.return_xcols(test)

        ytrain, ytest = self.return_ycols(train), self.return_ycols(test)
        self.save_split_index(split)

        return (Xtrain, Xtest, ytrain, ytest)
    
    def truncate(self, data_range):
        self.data = self.data.iloc[-data_range:]
            
