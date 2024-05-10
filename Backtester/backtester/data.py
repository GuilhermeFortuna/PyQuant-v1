from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np


class OHLC:
    def __init__(self, symbol: str, timeframe: str):
        """
        Initializes an instance of the OHLC class.

        This method sets the 'symbol' and 'timeframe' attributes to the provided values.
        It also initializes the 'data' attribute to None and the '_has_volume' attribute to None.

        Parameters:
        symbol (str): The symbol to be assigned to the instance.
        timeframe (str): The timeframe to be assigned to the instance.
        """

        self.symbol = symbol
        self.timeframe = timeframe

        # Initialize 'data' attribute to None. This will later hold the data for this instance.
        self.data = None

        # Initialize '_has_volume' attribute to None. This will later indicate whether the data includes a volume
        # column.
        self._has_volume = None

    @classmethod
    def group_by_date(cls, data: pd.DataFrame, symbol: str, timeframe: str):
        """
        Groups the provided data by date and creates an instance of the class for each group.

        This method takes a DataFrame, groups it by date, and then creates a new instance of the OHLC class for each
        group. The new instances have the grouped data assigned to their 'data' attribute and their values set as
        attributes.

        Parameters:
        data (pd.DataFrame): The DataFrame to be grouped by date.
        symbol (str): The symbol to be assigned to the new instances.
        timeframe (str): The timeframe to be assigned to the new instances.

        Yields:
        OHLC: An instance of the OHLC class for each group in the data.
        """

        # Group the data by date.
        groups = data.groupby(data.index.date)

        # Iterate over the groups.
        for _, group in groups:

            # Create a new instance of the class for each group.
            instance = cls(symbol=symbol, timeframe=timeframe)

            # Assign the group data to the 'data' attribute of the instance.
            instance.data = group

            # Set the values of the group data as attributes of the instance.
            instance._set_values_as_attributes()

            # Yield the instance.
            yield instance

    def _set_values_as_attributes(self):
        """
        Sets the values of the DataFrame as attributes of the instance.

        This method iterates over the columns of the 'data' DataFrame and sets each column's values as an attribute of
        the instance. The attribute's name is the same as the column's name. The 'datetime_index' and 'index'
        attributes are also set.

        Note: This method is intended to be used internally and may not provide the expected results if used outside
        of its intended context.
        """

        # Set the DataFrame's index as the 'datetime_index' attribute.
        self.datetime_index = self.data.index

        # Create an array of integers from 0 to the length of the DataFrame and set it as the 'index' attribute.
        self.index = np.array(range(len(self.data)), dtype=int)

        # Iterate over the columns of the DataFrame.
        for col in self.data.columns:
            # Set each column's values as an attribute of the instance. The attribute's name is the same as the column's
            # name.
            setattr(self, col, self.data[col].values)

    def load_data_from_csv(
            self,
            filepath: str,
            return_data: bool = False,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None,
            datetime_format: str = 'mixed',
            dayfirst: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Loads data from a CSV file and assigns it to the 'data' attribute of the instance.

        This method reads a CSV file, checks if it has the required columns, and converts the 'datetime' column to
        datetime format. It also checks if the CSV file contains a 'volume' column and sets the '_has_volume' attribute
        accordingly. If the numbers in the CSV file use comma as a decimal separator, it replaces it with a dot.
        The data is then sorted by index, sliced according to the provided dates, and assigned to the 'data' attribute.

        Parameters:
        filepath (str): The path to the CSV file.
        return_data (bool, optional): Whether to return the data. Defaults to False.
        date_from (datetime, optional): The start date for slicing the data. Defaults to None.
        date_to (datetime, optional): The end date for slicing the data. Defaults to None.
        datetime_format (str, optional): The format of the 'datetime' column in the CSV file. Defaults to 'mixed'.
        dayfirst (bool, optional): Whether the day is the first number in the date. Defaults to True.

        Returns:
        pd.DataFrame, optional: The loaded data, if 'return_data' is True. Otherwise, None.

        Raises:
        ValueError: If the CSV file does not have the required columns.
        """

        # Read CSV file.
        data = pd.read_csv(filepath)

        # Check if the CSV file has the required columns.
        if not data.columns.isin(['datetime', 'open', 'high', 'low', 'close']).sum() == 5:
            raise ValueError('The CSV file must have at least columns "datetime", "open", "high", "low", and "close".')

        # Check if CSV file contains volume column.
        if 'volume' in data.columns:
            self._has_volume = True

        # Convert datetime column to datetime and set as index.
        data['datetime'] = pd.to_datetime(data['datetime'], format=datetime_format, dayfirst=dayfirst)
        data.set_index('datetime', inplace=True)

        # Check if the numbers use comma as decimal separator and replace it with a dot.
        if ',' in data['open'].iat[0]:
            data = data.astype(str).apply(lambda x: x.str.replace(',', '.')).astype(float)
        else:
            data = data.astype({'open': float, 'high': float, 'low': float, 'close': float})

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
