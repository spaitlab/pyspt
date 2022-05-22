import numpy as np
import pandas as pd


def hampel_filter_with_dev_df(df: pd.DataFrame, vals_col: str, time_col=None, win_size=30, num_dev=3,
                              center_win=True) -> pd.DataFrame:
    """
    This function takes in dataframe containing time series of values, applies Hampel filter on
    these values, and returns dataframe consisting of original values columns along with
    the Hampel filtered data, outlier values, boolean flags where outliers found, values for lower
    deviation from median, values for upper deviation from median.

    Parameters
    ----------
    df: pd.DataFrame
        data from containing time series that needs to be Hampel filtered
    vals_col: str
        Single column name that contains values that need to be filtered.
    time_col: str
        Name of column that contains dates or timestamps
    win_size: int
        Size of sliding window for filtering.  Essentially the number of time steps to be considered when filtering.
    num_dev: int
        Number of standard deviations to consider when detecting values that would be considered outliers.
    center_win: Boolean
        Boolean value that determines whether the window is centered about the point being filtered?  Default=True.
        If False, point is at the leading edge (i.e. right side) of window  calculation.

    Returns
    -------
    Function returns a full dataframe consisting of original values columns along with
    the Hampel filtered data, outlier values and boolean flags where outliers found.
    """

    if (time_col != None):
        if (time_col not in list(df.columns)):
            raise Exception("Timestamp column '{}' is missing!".format(time_col))
        elif (time_col in list(df.columns)):
            if (not np.issubdtype(df[time_col].dtype, np.datetime64)):
                if (not np.issubdtype(pd.to_datetime(df[time_col]).dtype, np.datetime64)):
                    raise Exception("Timestamp column '{}' is not np.datetime64".format(time_col))
                else:
                    df[time_col] = pd.to_datetime(df[time_col])
                    drop_cols = set(df.columns) - set([time_col, vals_col])
                    # Not really filtered at this point. Just naming appropriately ahead of time.
                    orig_vals = df.sort_values(time_col, ascending=True).set_index(time_col).copy()
                    filtered = orig_vals.drop(columns=drop_cols).copy()
            else:
                df[time_col] = pd.to_datetime(df[time_col])
                drop_cols = set(df.columns) - set([time_col, vals_col])
                # Not really filtered at this point. Just naming appropriately ahead of time.
                orig_vals = df.sort_values(time_col, ascending=True).set_index(time_col).copy()
                filtered = orig_vals.drop(columns=drop_cols).copy()

    elif (time_col == None):
        if (not isinstance(df.index, pd.DatetimeIndex)):
            raise Exception("DataFrame index is not pd.DatetimeIndex")
        else:
            df.sort_index(inplace=True)
            drop_cols = set(df.columns) - set([vals_col])
            orig_vals = df.copy()
            filtered = orig_vals.drop(columns=drop_cols).copy()

    # Scale factor for estimating standard deviation based upon median value
    L = 1.4826

    # Calculate rolling median for the series
    rolling_median = filtered.rolling(window=int(win_size), center=center_win, min_periods=1).median()

    # Define a lambda function to apply to the series to calculate Median Absolute Deviation
    MAD = lambda x: np.median(np.abs(x - np.median(x)))

    # Calculate rolling MAD series
    rolling_MAD = filtered.rolling(window=(win_size), center=center_win, min_periods=1).apply(MAD)

    # Calculate threshold level for filtering based upon the number of standard deviation and
    # constant scaling factor L.
    threshold = int(num_dev) * L * rolling_MAD

    # Difference between original values and rolling median
    # Again, "filtered" not yet filtered at this point.
    difference = np.abs(filtered - rolling_median)

    median_minus_threshold = rolling_median - threshold
    median_minus_threshold.rename(columns={vals_col: 'LOWER_DEV'}, inplace=True)
    median_plus_threshold = rolling_median + threshold
    median_plus_threshold.rename(columns={vals_col: 'UPPER_DEV'}, inplace=True)

    '''
    # TODO: Look at logic here to possibly not mark as an outlier if threshold value
    is 0.0
    '''

    # Flag outliers
    outlier_idx = difference > threshold

    # Now it's filtered.  This should replace original values with filtered values from the rolling_median
    # dataframe where outliers were found.
    filtered[outlier_idx] = rolling_median[outlier_idx]
    filtered.rename(columns={vals_col: 'FLTRD_VAL'}, inplace=True)

    # Capture outliers column
    outliers = orig_vals[outlier_idx].rename(columns={vals_col: 'OUTLIER_VAL'}).drop(columns=drop_cols)
    # Capture outlier IS_OUTLIER column
    outlier_idx.rename(columns={vals_col: 'IS_OUTLIER'}, inplace=True)

    # The following returns a full dataframe consisting of original values columns
    # along with the Hampel filtered data, outlier values and boolean flags where outliers found.
    return pd.concat([orig_vals, filtered, outliers, outlier_idx, median_minus_threshold, median_plus_threshold], axis=1)
