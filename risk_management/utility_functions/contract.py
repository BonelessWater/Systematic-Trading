import asyncio
from enum import StrEnum
from dotenv import load_dotenv
import os
from pathlib import Path

import databento as db
import pandas as pd # type: ignore

load_dotenv()

# TODO: Add more vendor catalogs such as Norgate
class CATALOG(StrEnum):
    DATABENTO = f"data/catalog/databento/"
    NORGATE = f"data/catalog/norgate/"


class URI(StrEnum):
    DATABENTO = "s3://algogatorsbucket/catalog/databento/"
    NORGATE = "s3://algogatorsbucket/catalog/norgate/"


class ASSET(StrEnum):
    FUT = "FUT"
    OPT = "OPT"
    EQ = "EQ"


# TODO: Add more datasets
class DATASET(StrEnum):
    GLOBEX = "GLBX.MDP3"

    @classmethod
    def from_str(cls, value: str) -> "DATASET":
        """
        Converts a string to a DATASET enum based on the value to the Enum name and not value
        so "GLOBEX" -> DATASET.GLOBEX

        Args:
            - value: str - The value to convert to a DATASET enum

        Returns:
            - DATASET: The DATASET enum
        """
        try:
            return cls[value.upper()]
        except ValueError:

            for member in cls:
                if member.name.lower() == value.lower():
                    return member

            raise ValueError(f"{value} is not a valid {cls.__name__}")


# TODO: Add more schemas
class Agg(StrEnum):
    DAILY = "ohlcv-1d"
    HOURLY = "ohlcv-1h"
    MINUTE = "ohlcv-1m"
    SECOND = "ohlcv-1s"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class RollType(StrEnum):
    CALENDAR = "c"
    OPEN_INTEREST = "n"
    VOLUME = "v"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ContractType(StrEnum):
    FRONT = "0"
    BACK = "1"
    THIRD = "2"
    FOURTH = "3"
    FIFTH = "4"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class Contract:
    """
    Bar class to act as a base class for all bar classes

    Attributes:
    -   instrument: str - The instrument_id of the bar
    -   schema: Schema.BAR - The schema of the bar
    -   catalog: CATALOG - The catalog location of existing instrument data
    -   data: db.DBNStore - The data of the bar
    -   definitions: db.DBNStore - The definitions of the bar
    -   timestamp: pd.Timestamp - The timestamp of the bar
    -   open: pd.Series - The open price of the bar
    -   high: pd.Series - The high price of the bar
    -   low: pd.Series - The low price of the bar
    -   close: pd.Series - The close price of the bar
    -   volume: pd.Series - The volume of the bar
    -   expiration: pd.Series - The expiration of the bar
    -   instrument_id: pd.Series - The instrument_id of the bar

    Methods:

    GETTERS:
    -   get_timestamp() -> pd.Series - Returns the timestamp of the bar as a series
    -   get_open() -> pd.Series - Returns the open price of the bar as a series
    -   get_high() -> pd.Series - Returns the high price of the bar as a series
    -   get_low() -> pd.Series - Returns the low price of the bar as a series
    -   get_close() -> pd.Series - Returns the close price of the bar as a series
    -   get_volume() -> pd.Series - Returns the volume of the bar as a series
    -   get_bar() -> pd.DataFrame - Returns the bar as a dataframe
    -   get_expiration() -> pd.Series - Returns the expiration of the bar as a series
    -   get_instrument_id() -> pd.Series - Returns the instrument_id of the bar as a series

    CONSTRUCTORS:
    -   construct() -> None - Constructs the bar by first attempting to retrieve the data and definitions from the data catalog
    """

    # TODO: Refactor to use the @property decorator for getters and setters

    def __init__(
        self,
        instrument: str,
        dataset: DATASET,
        schema: Agg,
        catalog: CATALOG = CATALOG.DATABENTO,
    ):
        self.data: pd.DataFrame
        self.definitions: pd.DataFrame
        self._timestamp: pd.Index
        self._open: pd.Series
        self._high: pd.Series
        self._low: pd.Series
        self._close: pd.Series
        self._volume: pd.Series
        self._expiration: pd.Series
        self._instrument_id: pd.Series
        self._backadjusted: pd.Series = pd.Series()
        self.instrument: str = instrument
        self.dataset: DATASET = dataset
        self.schema: Agg = schema
        self.catalog: CATALOG = catalog

    def __str__(self) -> str:
        return f"Bar: {self.instrument} - {self.dataset} - {self.schema}"

    def __repr__(self) -> str:
        return f"Bar: {self.instrument} - {self.dataset} - {self.schema}"
    
    async def construct_async(
        self, client: db.Historical, roll_type: RollType, contract_type: ContractType
    ) -> None:
        """
        Asynchronously constructs the bar by first attempting to retrieve the data and definitions from the data catalog

        Args:
        -   client: db.Historical - The client to use to retrieve the data and definitions
        -   roll_type: RollType - The roll type of the bar
        -   contract_type: ContractType - The contract type of the bar

        Returns:
        None
        """
        data_path: Path = Path(
            f"{self.catalog}/{self.instrument}/{self.schema}/{roll_type}-{contract_type}-data.parquet"
        )
        definitions_path: Path = Path(
            f"{self.catalog}/{self.instrument}/{self.schema}/{roll_type}-{contract_type}-definitions.parquet"
        )

        range: dict[str, str] = await self._get_dataset_range_async(client)
        # Shift the data and definitions end back by one day to account for historical vs intraday data availability
        start: pd.Timestamp = pd.Timestamp(range["start"])
        end: pd.Timestamp = pd.Timestamp(range["end"]) - pd.Timedelta(days=1)

        if data_path.exists() and definitions_path.exists():
            try:
                self.data = pd.read_parquet(data_path)
                self.definitions = pd.read_parquet(definitions_path)
            except Exception as e:
                print(f"Error: {e}")
                return

            data_end: pd.Timestamp = pd.Timestamp(self.data.index[-1])
            # Check if the data and definitions are up to date
            if data_end != end:
                print(f"Data and Definitions are not up to date for {self.instrument}")
                await self._update_data_async(client, roll_type, contract_type, data_end, end)
        else:
            print(f"Data and Definitions not present for {self.instrument}")
            await self._fetch_initial_data_async(client, roll_type, contract_type, start, end)

        # Save the new data and definitions to the catalog
        self._save_data(data_path, definitions_path)

        # Set the timestamp, open, high, low, close, and volume
        self._set_attributes()


    async def _get_dataset_range_async(self, client: db.Historical) -> dict[str, str]:
        return await asyncio.to_thread(client.metadata.get_dataset_range, dataset=self.dataset)

    async def _update_data_async(self, client: db.Historical, roll_type: RollType, contract_type: ContractType, data_end: pd.Timestamp, end: pd.Timestamp):
        try:
            symbols: str = f"{self.instrument}.{roll_type}.{contract_type}"
            # Add one day to the end as the stream request is end exclusive
            end += pd.Timedelta(days=1)
            new_data: db.DBNStore = await self._fetch_databento_data_async(client, symbols, data_end, end) 
            new_definitions: db.DBNStore = await self._fetch_databento_definitions_async(client, new_data)
            
            # Combine new data with existing data and skip duplicates if they exist based on index
            self.data = pd.concat([self.data, new_data.to_df()])
            self.data = self.data[~self.data.index.duplicated(keep="last")]
            self.definitions = pd.concat([self.definitions, new_definitions.to_df()])
            self.definitions = self.definitions[~self.definitions.index.duplicated(keep="last")]
        except Exception as e:
            print(f"Error: {e}")

    async def _fetch_initial_data_async(self, client: db.Historical, roll_type: RollType, contract_type: ContractType, start: pd.Timestamp, end: pd.Timestamp):
        symbols: str = f"{self.instrument}.{roll_type}.{contract_type}"
        data: db.DBNStore = await self._fetch_databento_data_async(client, symbols, start, end)
        definitions: db.DBNStore = await self._fetch_databento_definitions_async(client, data)

        self.data = data.to_df()
        self.definitions = definitions.to_df()

    async def _fetch_databento_data_async(self, client: db.Historical, symbols: str, start: pd.Timestamp, end: pd.Timestamp) -> db.DBNStore:
        return await asyncio.to_thread(
            client.timeseries.get_range,
            dataset=str(self.dataset),
            symbols=[symbols],
            schema=db.Schema.from_str(self.schema),
            start=start,
            end=end,
            stype_in=db.SType.CONTINUOUS,
            stype_out=db.SType.INSTRUMENT_ID,
        )

    async def _fetch_databento_definitions_async(self, client: db.Historical, data: db.DBNStore) -> db.DBNStore:
        return await asyncio.to_thread(data.request_full_definitions, client)

    def _set_attributes(self):
        self.timestamp = self.data.index
        self.open = pd.Series(self.data["open"])
        self.high = pd.Series(self.data["high"])
        self.low = pd.Series(self.data["low"])
        self.close = pd.Series(self.data["close"])
        self.volume = pd.Series(self.data["volume"])
        self.instrument_id = pd.Series(self.definitions["instrument_id"])
        try:
            self.expiration = self._set_exp(self.data.copy(), self.definitions.copy())
        except Exception as e:
            print(f"Error within converting expiration into daily data: {e}")

    def _save_data(self, data_path: Path, definitions_path: Path):
        data_path.parent.mkdir(parents=True, exist_ok=True)
        definitions_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(data_path)
        self.definitions.to_parquet(definitions_path)

    @property
    def timestamp(self) -> pd.Index:
        """
        Returns the timestamp of the bar as a series

        Args:
        None

        Returns:
        pd.Index: The timestamp of the bar as a pandas index
        """
        if self._timestamp.empty:
            raise ValueError("Timestamp is empty")
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: pd.Index) -> None:
        """
        Sets the timestamp of the bar as a series

        Args:
        value: pd.Index - The timestamp of the bar as a pandas index

        Returns:
        None
        """
        self._timestamp = value

    @property
    def open(self) -> pd.Series:
        """
        Returns the open price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The open price of the bar as a series
        """
        if self._open.empty:
            raise ValueError("Open is empty")
        return self._open

    @open.setter
    def open(self, value: pd.Series) -> None:
        """
        Sets the open price of the bar as a series

        Args:
        value: pd.Series - The open price of the bar as a series

        Returns:
        None
        """
        self._open = value

    @property
    def high(self) -> pd.Series:
        """
        Returns the high price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The high price of the bar as a series
        """
        if self._high.empty:
            raise ValueError("High is empty")
        return self._high

    @high.setter
    def high(self, value: pd.Series) -> None:
        """
        Sets the high price of the bar as a series

        Args:
        value: pd.Series - The high price of the bar as a series

        Returns:
        None
        """
        self._high = value

    @property
    def low(self) -> pd.Series:
        """
        Returns the low price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The low price of the bar as a series
        """
        if self._low.empty:
            raise ValueError("Low is empty")
        return self._low

    @low.setter
    def low(self, value: pd.Series) -> None:
        """
        Sets the low price of the bar as a series

        Args:
        value: pd.Series - The low price of the bar as a series

        Returns:
        None
        """
        self._low = value

    @property
    def close(self) -> pd.Series:
        """
        Returns the close price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The close price of the bar as a series
        """
        if self._close.empty:
            raise ValueError("Close is empty")
        return self._close

    @close.setter
    def close(self, value: pd.Series) -> None:
        """
        Sets the close price of the bar as a series

        Args:
        value: pd.Series - The close price of the bar as a series

        Returns:
        None
        """
        self._close = value

    @property
    def volume(self) -> pd.Series:
        """
        Returns the volume of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The volume of the bar as a series
        """
        if self._volume.empty:
            raise ValueError("Volume is empty")
        return self._volume

    @volume.setter
    def volume(self, value: pd.Series) -> None:
        """
        Sets the volume of the bar as a series

        Args:
        value: pd.Series - The volume of the bar as a series

        Returns:
        None
        """
        self._volume = value

    @property
    def expiration(self) -> pd.Series:
        """
        Returns the expiration of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The expiration of the bar as a series
        """
        if self._expiration.empty:
            raise ValueError("Expiration is empty")
        return self._expiration

    @expiration.setter
    def expiration(self, value: pd.Series) -> None:
        """
        Sets the expiration of the bar as a series

        Args:
        value: pd.Series - The expiration of the bar as a series

        Returns:
        None
        """
        self._expiration = value

    @property
    def instrument_id(self) -> pd.Series:
        """
        Returns the instrument_id of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The instrument_id of the bar as a series
        """
        if self._instrument_id.empty:
            raise ValueError("Instrument ID is empty")
        return self._instrument_id

    @instrument_id.setter
    def instrument_id(self, value: pd.Series) -> None:
        """
        Sets the instrument_id of the bar as a series

        Args:
        value: pd.Series - The instrument_id of the bar as a series

        Returns:
        None
        """
        self._instrument_id = value

    def get_instrument(self) -> str:
        """
        Returns the instrument_id of the bar

        Args:
        None

        Returns:
        str: The instrument_id of the bar
        """
        return self.instrument

    def get_dataset(self) -> DATASET:
        """
        Returns the dataset of the bar

        Returns:
        DATASET: The dataset of the bar
        """
        return self.dataset

    def get_schema(self) -> Agg:
        """
        Returns the schema of the bar

        Returns:
        Schema.BAR: The schema of the bar
        """
        return self.schema

    def get_catalog(self) -> CATALOG:
        """
        Returns the catalog location of the existing instrument data

        Returns:
        CATALOG: The catalog location of the existing instrument data
        """
        return self.catalog

    def get_timestamp(self) -> pd.Index:
        """
        Returns the timestamp of the bar as a series

        Returns:
        pd.Index: The timestamp of the bar as a pandas index
        """
        if self.timestamp.empty:
            raise ValueError("Timestamp is empty")
        return self.timestamp

    def get_open(self) -> pd.Series:
        return self.open

    def get_high(self) -> pd.Series:
        return self.high

    def get_low(self) -> pd.Series:
        return self.low

    def get_close(self) -> pd.Series:
        return self.close

    def get_volume(self) -> pd.Series:
        return self.volume

    def get_expiration(self) -> pd.Series:
        return self.expiration

    def get_instrument_id(self) -> pd.Series:
        """
        Returns the instrument_id of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The instrument_id of the bar as a series
        """
        if self.instrument_id.empty:
            raise ValueError("Instrument ID is empty")
        return self.instrument_id

    def get_contract(self) -> pd.DataFrame:
        """
        Returns the bar as a dataframe

        Args:
        None

        Returns:
        pd.DataFrame: The bar as a dataframe
        """
        return pd.DataFrame(
            {
                "timestamp": self.get_timestamp(),
                "open": self.get_open(),
                "high": self.get_high(),
                "low": self.get_low(),
                "close": self.get_close(),
                "volume": self.get_volume(),
            }
        )

    @property
    def open_interest(self) -> pd.Series:
        """
        Returns the open interest of the bar as a series

        ONLY VALID FOR NORGATE DATA

        Args:
        None

        Returns:
        pd.Series: The open interest of the bar as a series
        """
        if self.catalog == CATALOG.NORGATE:
            if self.data.empty:
                raise ValueError("Data is empty")
            else:
                return self._open_interest
        elif self.catalog == CATALOG.DATABENTO:
            raise ValueError("Open Interest is not present in Databento Data")

    def _perform_backadjustment(self, data: pd.DataFrame) -> pd.Series:
        """
        Perform backadjustment on the close prices of the data

        Args:
        - data: pd.DataFrame - The data to perform backadjustment on containing the instrument_id and close prices

        Returns:
        - pd.Series: The backadjusted data
        """

        backadjusted: pd.DataFrame = data.sort_index(ascending=False)
        cum: float = 0.0
        adj: float = 0.0
        for i in range(1, len(backadjusted)):
            if (
                backadjusted.iloc[i]["instrument_id"]
                != backadjusted.iloc[i - 1]["instrument_id"]
            ):
                adj = backadjusted["close"].iloc[i - 1] - backadjusted["close"].iloc[i]
                cum += adj
                backadjusted.loc[backadjusted.index[i] :, "close"] += adj

        backadjusted.sort_index(ascending=True, inplace=True)
        return pd.Series(backadjusted["close"])

    @property
    def backadjusted(self) -> pd.Series:
        """
        Returns the backadjusted series as a pandas series

        BACKADJUSTMENT:
        Backadjustment is the process of removing sudden gaps in pricing data when a contracts rolls over from the front to the back.
        While it does not preserve the integrity of the data, it gives a better representation of true pricemovements.
        The process goes as follows:
            1. Iterate backwards through a frame of both pricing data, and expiration(or another alias) data.
            2. When a contract switches to the previous contract, calculate the difference between the current contract and previous contract.
            3. Apply that previous adjustment to the current contract and all other following contracts keeping a rolling value
            4. Repeat!

        This method of futures adjustment is know as the panama canal method

        Args:
        None

        Returns:
        pd.DataFrame: The backadjusted bar as a dataframe
        """
        if self.catalog == CATALOG.NORGATE:
            if self.data.empty:
                raise ValueError("Data is empty")
            else:
                return self._backadjusted

        elif self.catalog == CATALOG.DATABENTO:
            if self.definitions.empty or self.data.empty:
                raise ValueError("Data and Definitions are not present")
            elif (
                self.open.empty
                or self.high.empty
                or self.low.empty
                or self.close.empty
                or self.volume.empty
            ):
                raise ValueError("Open, High, Low, Close, or Volume is empty")
            elif self._backadjusted.empty:
                # Perform backadjustment on close prices and return the backadjusted series
                # TODO: Check logic for backadjustment
                backadjusted: pd.DataFrame = pd.DataFrame(
                    self.data.copy()[["close", "instrument_id"]]
                )
                tmp_backadjusted: pd.Series = self._perform_backadjustment(backadjusted)
                self._backadjusted = tmp_backadjusted
                return self._backadjusted
            else:
                return self._backadjusted

    @backadjusted.setter
    def backadjusted(self, value: pd.Series) -> None:
        """
        Setter for backadjusted series

        Args:
        -   value: pd.Series | A pandas Series of backadjusted prices

        Return:
        -   None
        """
        self._backadjusted = value

    def _set_exp(self, data: pd.DataFrame, definitions: pd.DataFrame) -> pd.Series:
        """
        This _set_exp method is used to create an the correct timeseries that follows the the daily data of the OHLCV data versus the sparse data of the defintions which contains the correct expiration

        Args:
            - instrument_ids: pd.Series - The daily individual instrument_id
            - definitions: pd.Series - The definitions of our instruments which include both our expirations as well as matching instrument_id to our data series.
        """
        # We need to index our definitons by the instrument_id
        exp_df: pd.DataFrame = (
            definitions.reset_index()[["expiration", "instrument_id"]]
            .drop_duplicates()
            .set_index("instrument_id")
        )

        exp_df = exp_df[~exp_df.index.duplicated(keep="first")]
        # We then need to map our instrument_ids to the correct expiration date using the definitions while preserving the data frame index
        data["expiration"] = data["instrument_id"].map(exp_df["expiration"])
        # Finally we need to set the index of our instrument ids to the same index as our data using the timestamp
        expirations: pd.Series = data["expiration"]
        return expirations

    def construct_norgate(self) -> None:
        """
        Constructs the contract using the Norgate catalog and data which contains historical data ranging farther back than Databento

        Norgate Data Format:
        - index: numeric - The index of the bar
        - Date: daily date in the format MM/DD/YYYY
        - Security Name: str - The name of the security
        - Open: float - The open price of the bar
        - High: float - The high price of the bar
        - Low: float - The low price of the bar
        - Close: float - The close price of the bar
        - Volume: int - The volume of the bar
        - Open Interest: int - The open interest of the bar
        - Delivery Month: int - The delivery month of the bar in the format YYYYMM
        - Time: time in the format HH:MM:SS
        - Unadj_Close: float - The unadjusted close price of the bar
        - Next Quarter: int - The next quarter of the bar in the format YYYYMM

        Args:
        -   contract_type: ContractType - The contract type of the bar

        Returns:
        -   None
        """
        data_path_csv: Path = Path(f"{self.catalog}/_{self.instrument}_Data.csv")
        data_path_parquet: Path = Path(
            f"{self.catalog}/_{self.instrument}_Data.parquet"
        )

        # Check if the data exists
        if data_path_csv.exists():
            try:
                self.data = pd.read_csv(data_path_csv)
            except Exception as e:
                print(f"Could not read data: {e}")
                raise

            # Combine the Date and Time columns to create a timestamp
            self.data["timestamp"] = pd.to_datetime(
                self.data["Date"] + " " + self.data["Time"]
            )

            # Set the timestamp to the index
            self.data.set_index("timestamp", inplace=True)

            # Set the timestamp, open, high, low, unadj_close, volume, and open interest
            self.timestamp = self.data.index
            self.open = pd.Series(self.data["Open"])
            self.high = pd.Series(self.data["High"])
            self.low = pd.Series(self.data["Low"])
            self.close = pd.Series(self.data["Unadj_Close"])
            self.volume = pd.Series(self.data["Volume"])
            self._open_interest = pd.Series(self.data["Open Interest"])
            # Instrument ID becomes the delivery month
            self.instrument_id = pd.Series(self.data["Delivery Month"])
            # Expiration becomes the Datetime of the Delivery Month
            self.expiration = pd.to_datetime(self.data["Delivery Month"], format="%Y%m")
            # Set the backadjusted series to the close prices
            self._backadjusted = pd.Series(self.data["Close"])

            # Write to the catalog using parquet format and save the data
        else:
            raise (
                ValueError(
                    f"Data not present for {self.instrument}. Please check the catalog path {self.catalog}"
                )
            )

    def construct(
        self, client: db.Historical, roll_type: RollType, contract_type: ContractType
    ) -> None:
        """
        Constructs the bar by first attempting to retrieve the data and definitions from the data catalog

        Args:
        -   client: db.Historical - The client to use to retrieve the data and definitions
        -   roll_type: RollType - The roll type of the bar
        -   contract_type: ContractType - The contract type of the bar

        Returns:
        None
        """
        data_path: Path = Path(
            f"{self.catalog}/{self.instrument}/{self.schema}/{roll_type}-{contract_type}-data.parquet"
        )
        definitions_path: Path = Path(
            f"{self.catalog}/{self.instrument}/{self.schema}/{roll_type}-{contract_type}-definitions.parquet"
        )

        range: dict[str, str] = client.metadata.get_dataset_range(dataset=self.dataset)
        # Shift the data and definitions end back by one day to account for historical vs intraday data availability
        start: pd.Timestamp = pd.Timestamp(range["start"]) - pd.Timedelta(days=1)
        end: pd.Timestamp = pd.Timestamp(range["end"]) - pd.Timedelta(days=1)

        if data_path.exists() and definitions_path.exists():
            try:
                self.data = pd.read_parquet(data_path)
                self.definitions = pd.read_parquet(definitions_path)
            except Exception as e:
                print(f"Error: {e}")
                return

            data_end: pd.Timestamp = pd.Timestamp(self.data.index[-1])
            definitions_end: pd.Timestamp = pd.Timestamp(self.definitions.index[-1])
            # Check if the data and definitions are up to date
            if data_end != end:
                print(f"Data and Definitions are not up to date for {self.instrument}")
                # Try to retrieve the new data and definitions but if failed then do not update
                try:
                    symbols: str = f"{self.instrument}.{roll_type}.{contract_type}"
                    new_data: db.DBNStore = client.timeseries.get_range(
                        dataset=self.dataset,
                        symbols=[symbols],
                        schema=db.Schema.from_str(self.schema),
                        start=data_end,
                        end=end,
                        stype_in=db.SType.CONTINUOUS,
                        stype_out=db.SType.INSTRUMENT_ID,
                    )
                    new_definitions: db.DBNStore = new_data.request_full_definitions(
                        client=client
                    )
                    # Combine new data with existing data and skip duplicates if they exist based on index
                    self.data = pd.concat(
                        [self.data, new_data.to_df()]
                    )
                    self.data = self.data[~self.data.index.duplicated(keep="last")]
                    self.definitions = pd.concat(
                        [self.definitions, new_definitions.to_df()]
                    )
                    self.definitions = self.definitions[~self.definitions.index.duplicated(keep="last")]
                except Exception as e:
                    print(f"Error: {e}")

            # Save the new data and definitions to the catalog
            self.data.to_parquet(data_path)
            self.definitions.to_parquet(definitions_path)

            # Set the timestamp, open, high, low, close, and volume
            self.timestamp = self.data.index
            self.open = pd.Series(self.data["open"])
            self.high = pd.Series(self.data["high"])
            self.low = pd.Series(self.data["low"])
            self.close = pd.Series(self.data["close"])
            self.volume = pd.Series(self.data["volume"])
            self.instrument_id = pd.Series(self.definitions["instrument_id"])
            self.expiration = self._set_exp(self.data.copy(), self.definitions.copy())

        else:
            print(f"Data and Definitions not present for {self.instrument}")
            print(f"Attempting to retrieve data and definitions for {self.instrument}")
            print(f"Creating data and definitions for {self.instrument} at {data_path}")
            # Submit a job request to retrieve the data and definitions
            symbols: str = f"{self.instrument}.{roll_type}.{contract_type}"
            # TODO: Implement job request submission
            # details: dict[str, Any] = client.batch.submit_job(dataset=self.dataset, symbols=symbols, schema=db.Schema.from_str(self.schema), encoding=db.Encoding.DBN start=start, end=end, stype_in=db.SType.CONTINUOUS, split_duration=db.SplitDuration.NONE)
            # print(f"Job Request Submitted: {details["symbols"]} - {details["schema"]} - {details["start"]} - {details["end"]}")
            data: db.DBNStore = client.timeseries.get_range(
                dataset=str(self.dataset),
                symbols=[symbols],
                schema=db.Schema.from_str(self.schema),
                start=start,
                end=end,
                stype_in=db.SType.CONTINUOUS,
                stype_out=db.SType.INSTRUMENT_ID,
            )
            definitions: db.DBNStore = data.request_full_definitions(client=client)

            # Make the directories if they do not exist
            data_path.parent.mkdir(parents=True, exist_ok=True)
            definitions_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the data and definitions to the catalog
            data.to_parquet(data_path)
            definitions.to_parquet(definitions_path)

            self.data = data.to_df()
            self.definitions = definitions.to_df()

            # Set the timestamp, open, high, low, close, and volume
            self.timestamp = self.data.index
            self.open = pd.Series(self.data["open"])
            self.high = pd.Series(self.data["high"])
            self.low = pd.Series(self.data["low"])
            self.close = pd.Series(self.data["close"])
            self.volume = pd.Series(self.data["volume"])
            self.instrument_id = pd.Series(self.definitions["instrument_id"])
            self.expiration = self._set_exp(self.data.copy(), self.definitions.copy())

            # WARN: The API "should" be able to handle data requests under 5 GB but have had issues in the pass with large requests
            return

async def main() -> None:
    contract: Contract = Contract(
        instrument="ES",
        dataset=DATASET.GLOBEX,
        schema=Agg.DAILY,
        catalog=CATALOG.DATABENTO,
    )
    client: db.Historical = db.Historical(os.getenv("DATABENTO_API_KEY"))
    task = contract.construct_async(client, RollType.CALENDAR, ContractType.FRONT)
    await task

    print(contract.get_contract())
    

if __name__ == "__main__":
    # Example Usage
    asyncio.run(main())
