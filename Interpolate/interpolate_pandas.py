import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
import scipy.interpolate

"""
 caused by float index, the two functions may differ a bit
    >>> xs = arange(0, 1, 0.1)
    >>> ys = cos(xs)
    >>> df = DataFrame({'x': xs, 'y': ys})
    >>> df2 = df.set_index('x')
    >>> df3 = df.reindex(arange(0, 1, 0.01))
    >>> df3.index[30] == df['x'][3]
    False
    >>> print df3.index[30], df['x'][3]
    0.3 0.3
    >>> df3.index[30]
    0.29999999999999999
    >>> df['x'][3]
    0.30000000000000004
"""
def pandas_interpolate(df, interp_column, method = 'cubic'):
    df = df.set_index(interp_column)
    # df = df.reindex(np.arange(df.index.min(),df.index.max(), 0.0005))
    at = np.arange(df.index.min(), df.index.max(), 0.0005)
    ## change to the union of new and old
    df = df.reindex(df.index | at)
    df = df.interpolate(method=method)
    df = df.reset_index()
    df = df.rename(columns = {'index':interp_column})
    return df

def scipy_interpolate(df, interp_column, method = 'cubic'):
    series = {}
    new_x = np.arange(df[interp_column].min(), df[interp_column].max(), 0.0005)
    for column in df:
        if column == interp_column:
            series[column] = new_x
        else:
            interp_f = scipy.interpolate.interp1d(df[interp_column], df[column],kind= method)
            series[column] = interp_f(new_x)
    return pd.DataFrame(series)

if __name__ == '__main__':
    # df = pd.read_csv('interp_test.csv')
    # pd_interp = pandas_interpolate(df, 'distance', 'cubic')
    # scipy_interp = scipy_interpolate(df, 'distance', 'cubic')
    #
    # ## plot
    # mlp.plot(pd_interp['lon'], pd_interp['lat'], label='pandas')
    # mlp.plot(scipy_interp['lon'], scipy_interp['lat'], label='scipy')
    # mlp.legend(loc = 'best')

    mlp.figure()
    df2 = pd.DataFrame({'x':np.arange(10), 'sin(x)':np.sin(np.arange(10))})
    pd_interp2 = pandas_interpolate(df2, 'x', 'cubic')
    scipy_interp2 = scipy_interpolate(df2, 'x', 'cubic')
    mlp.plot(pd_interp2['x'], pd_interp2['sin(x)'], label='pandas')
    mlp.plot(scipy_interp2['x'], scipy_interp2['sin(x)'], label='scipy')
    mlp.legend(loc = 'best')
    mlp.show()

