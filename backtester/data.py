from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np


class OHLC:

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe

        self.data = None
        self._has_volume = None

    def load_data_from_csv(
            self,
            filepath: str,
            return_data: bool = False,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Loads data from a CSV file.

        This method reads a CSV file from the provided filepath and processes it to ensure it contains the required columns.
        It also checks if the CSV file contains a 'volume' column and sets the '_has_volume' attribute accordingly.
        The 'datetime' column is converted to datetime format and set as the index of the DataFrame.
        The DataFrame is sorted by the index and sliced according to the provided 'date_from' and 'date_to' parameters.
        The processed DataFrame is assigned to the 'data' attribute of the instance.

        Parameters:
        filepath (str): The path to the CSV file.
        return_data (bool, optional): If True, the method returns the loaded data. Defaults to False.
        date_from (datetime, optional): The start date for slicing the data. If not provided, the first date in the data is used.
        date_to (datetime, optional): The end date for slicing the data. If not provided, the last date in the data is used.

        Returns:
        pd.DataFrame: The loaded and processed data if 'return_data' is True. Otherwise, None.

        Raises:
        ValueError: If the CSV file does not contain the required columns.
        """

        # Read CSV file.
        data = pd.read_csv(filepath)

        # Check if the CSV file has the required columns.
        if not data.columns.isin(['datetime', 'open', 'high', 'low', 'close']).sum == 4:
            raise ValueError('The CSV file must have at least columns "datetime", "open", "high", "low", and "close".')

        # Check if CSV file contains volume column.
        if 'volume' in data.columns:
            self._has_volume = True

        # Convert datetime column to datetime and set as index.
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)

        # Sort data by index.
        if not data.index.is_monotonic_increasing:
            data.sort_index(inplace=True)

        # Define date_from and date_to, if not provided.
        if date_from is None:
            date_from = data.index[0]

        if date_to is None:
            date_to = data.index[-1]

        # Slice data and assign to 'data' attribute.
        self.data = data.loc[date_from:date_to]

        # Return data, if requested.
        if return_data:
            return self.data
