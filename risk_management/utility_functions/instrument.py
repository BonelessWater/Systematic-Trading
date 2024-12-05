import os
from typing import Tuple, Optional
import asyncio
import databento as db
import pandas as pd
from enum import Enum
from dotenv import load_dotenv
from risk_management.utility_functions.contract import (
    ASSET,
    DATASET,
    CATALOG,
    Agg,
    RollType,
    Contract,
    ContractType,
)

load_dotenv()

class SecurityType(Enum):
    FUTURE = ('Future', 'FUT')

    def __init__(self, obj_name : str, string : str):
        self._obj_name : str = obj_name
        self.string : str = string

    @property
    def obj(self):
        # Dynamically resolve the object class when accessed
        if isinstance(self._obj_name, str):
            return globals()[self._obj_name]
        return self._obj_name

    @classmethod
    def from_str(cls, value: str) -> "SecurityType":
        """
        Converts a string to a SecurityType enum based on the value to the Enum name and not value
        so "FUTURE" -> FUTURE

        Args:
            - value: str - The value to convert to a SecurityType enum

        Returns:
            - SecurityType: The SecurityType enum
        """
        try:
            return cls[value.upper()]
        except ValueError:
            # If exact match fails, look for a case-insensitive match
            for member in cls:
                if member.name.lower() == value.lower():
                    return member

            raise ValueError(f"{value} is not a valid {cls.__name__}")

class Instrument():
    """
    Instrument class to act as a base class for all asset classes

    Attributes:
    symbol: str - The symbol of the instrument
    dataset: str - The dataset of the instrument

    Methods:
    get_symbol() -> str - Returns the symbol of the instrument
    get_dataset() -> str - Returns the dataset of the instrument
    get_collection() -> Tuple[str, str] - Returns the symbol and dataset of the instrument

    The instrument class is an

    """

    security_type: SecurityType

    def __init__(
            self,
            symbol: str,
            dataset: DATASET,
            currency : str,
            exchange : str,
            security_type: Optional['SecurityType'] = None,
            multiplier : float = 1.0,
            ib_symbol : str | None = None
        ) -> None:
        self._symbol = symbol
        self._ib_symbol = ib_symbol if ib_symbol is not None else symbol
        self._dataset = dataset
        self.client: db.Historical = db.Historical(os.getenv("DATABENTO_API_KEY"))
        self.multiplier = multiplier
        self._currency = currency
        self._exchange = exchange

        if security_type is not None:
            self.__class__ = security_type.obj

    @property
    def symbol(self) -> str:
        """
        Returns the symbol of the instrument

        Args:
        None

        Returns:
        str: The symbol of the instrument
        """
        return self._symbol

    @property
    def ib_symbol(self) -> str:
        """
        Returns the IBKR symbol of the instrument
        
        Args:
        None
        
        Returns:
        str: The IBKR symbol of the instrument
        """
        return self._ib_symbol

    @property
    def currency(self) -> str:
        """
        Returns the currency of the instrument

        Args:
        None

        Returns:
        str: The currency the instrument is denominated in
        """
        return self._currency
    
    @property
    def exchange(self) -> str:
        """
        Returns the exchange the instrument trades on

        Args:
        None

        Returns:
        str: The exchange the instrument trades on
        """
        return self._exchange

    @property
    def dataset(self) -> DATASET:
        """
        Returns the dataset of the instrument

        Args:
        None

        Returns:
        str: The dataset of the instrument
        """
        return self._dataset

    def get_symbol(self) -> str:
        """
        Returns the symbol of the instrument

        Args:
        None

        Returns:
        str: The symbol of the instrument
        """
        return self.symbol

    @property
    def name(self) -> str:
        """
        Returns the name of the instrument

        Args:
        None

        Returns:
        str: The name of the instrument
        """
        return self.symbol

    def get_dataset(self) -> str:
        """
        Returns the dataset of the instrument

        Args:
        None

        Returns:
        str: The dataset of the instrument
        """
        return self.dataset

    def get_collection(self) -> Tuple[str, str]:
        """
        Returns the symbol and dataset of the instrument

        Args:
        None

        Returns:
        Tuple[str, str]: The symbol and dataset of the instrument
        """
        return (self.symbol, self.dataset)

    #! PRICE MUST BE THE PRICE THAT YOU WANT TO USE FOR BACKTESTING
    @property
    def price(self) -> pd.Series:
        """
        Returns the prices of the instrument

        Args:
        None

        Returns:
        pd.Series: The prices of the instrument
        """
        raise NotImplementedError()

    @property
    def percent_returns(self) -> pd.Series:
        """
        Returns the percent returns of the instrument

        Args:
        None

        Returns:
        pd.Series: The percent returns of the instrument
        """
        raise NotImplementedError()

    async def add_data(
        self, agg: Agg, roll_type: RollType, contract_type: ContractType
    ) -> None:
        """
        Asynchronously add data to the instrument.
        """
        client = db.Historical(key=os.getenv("DATABENTO_KEY"))

        contract = Contract(
            instrument=self.symbol,
            dataset=self.dataset,
            schema=agg,
        )

        try:
            await contract.construct_async(client, roll_type, contract_type)
            self._process_and_store_data(contract, agg, roll_type, contract_type)
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")

    def _process_and_store_data(
        self,
        contract: Contract,
        agg: Agg,
        roll_type: RollType,
        contract_type: ContractType,
    ):
        # Process and store the fetched data
        if contract_type == ContractType.FRONT:
            self.front = contract
            self.price = contract.backadjusted
        elif contract_type == ContractType.BACK:
            self.back = contract
        # Add more conditions if needed for other roll types and contract types


class Future(Instrument):
    """
    Future class is a representation of a future instrument within the financial markets
    Within future contructs we can have multiple contracts that represent the same underlying asset so we need to be able to handle multiple contracts like a front month and back month contract
    To implement this we will have a list of contracts that the future instrument will handle

    Attributes:
    symbol: str - The symbol of the future instrument
    dataset: str - The dataset of the future instrument
    contracts: dict[str, Contract] - The contracts of the future instrument
    front: Contract - The front month contract of the future instrument
    back: Contract - The back month contract of the future instrument


    Methods:
    -   add_contract(contract: Contract, contract_type: ContractType) -> None - Adds a contract to the future instrument
    """

    security_type = SecurityType.FUTURE
    def __init__(self, symbol: str, dataset: DATASET, currency : str, exchange : str, multiplier: float = 1.0):
        super().__init__(
            symbol,
            dataset,
            currency,
            exchange,
            security_type=SecurityType.FUTURE,
            multiplier=multiplier
        )

        self.multiplier: float = multiplier
        self.contracts: dict[str, Contract] = {}
        self._front: Contract
        self._back: Contract
        self._price: pd.Series

    @property
    def front(self) -> Contract:
        """
        Returns the front month contract of the future instrument

        Args:
        None

        Returns:
        Bar: The front month contract of the future instrument
        """
        if not hasattr(self, "_front"):
            raise ValueError("Front is empty")
        return self._front

    @front.setter
    def front(self, value: Contract) -> None:
        """
        Sets the front month contract of the future instrument

        Args:
        value: Bar - The front month contract of the future instrument

        Returns:
        None
        """
        self._front = value

    @front.deleter
    def front(self) -> None:
        """
        Deletes the front month contract of the future instrument

        Args:
        None

        Returns:
        None
        """
        del self._front

    front.__doc__ = """
    The front month contract of the future instrument

    Args:
        
    Returns:
        pd.Series: The front month contract of the future instrument
    """

    @property
    def price(self) -> pd.Series:
        """
        Returns the price of the future instrument

        Args:
        None

        Returns:
        pd.Series: The price of the future instrument
        """
        if self._price.empty:
            raise ValueError("Price is empty")
        return self._price

    @price.setter
    def price(self, value: pd.Series) -> None:
        """
        Sets the price of the future instrument

        Args:
        value: pd.Series - The price of the future instrument

        Returns:
        None
        """
        self._price = value

    @price.deleter
    def price(self) -> None:
        """
        Deletes the price of the future instrument

        Args:
        None

        Returns:
        None
        """
        del self._price

    def __str__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def __repr__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def get_contracts(self) -> dict[str, Contract]:
        """
        Returns the contracts of the future instrument

        Args:
        None

        Returns:
        dict[str, Contract]: The contracts of the future instrument
        """
        if self.contracts == {}:
            raise ValueError("No Contracts are present")
        else:
            return self.contracts

    def get_front(self) -> Contract:
        return self.front

    def get_back(self) -> Contract:
        return self.back

    def add_data(
        self,
        schema: Agg,
        roll_type: RollType,
        contract_type: ContractType,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds data to the future instrument but first creates a bar object based on the schema

        Args:
        schema: Schema.BAR - The schema of the bar
        roll_type: RollType - The roll type of the bar
        contract_type: ContractType - The contract type of the bar
        name: Optional[str] - The name of the bar

        Returns:
        None
        """
        contract: Contract = Contract(
            instrument=self.symbol,
            dataset=self.dataset,
            schema=schema,
        )

        if name is None:
            name = f"{contract.get_instrument()}-{roll_type}-{contract_type}"

        contract.construct(
            client=self.client, roll_type=roll_type, contract_type=contract_type
        )

        self.contracts[name] = contract
        if contract_type == ContractType.FRONT:
            self.front = contract
            self.price = contract.backadjusted
        elif contract_type == ContractType.BACK:
            self.back = contract

    def add_norgate_data(self, name: Optional[str] = None) -> None:
        """
        Adds data to the future instrument but first creates a bar object based on the schema

        Args:
        name: Optional[str] - The name of the bar

        Returns:
        None
        """
        contract: Contract = Contract(
            instrument=self.symbol,
            dataset=self.dataset,
            schema=Agg.DAILY,
            catalog=CATALOG.NORGATE,
        )

        if name is None:
            name = f"{contract.get_instrument()}"

        contract.construct_norgate()

        self.contracts[name] = contract
        self.front = contract
        self.price = contract.backadjusted

    async def add_data_async(
        self,
        schema: Agg,
        roll_type: RollType,
        contract_type: ContractType,
        name: Optional[str] = None,
    ) -> None:
        """
        Asynchronously adds data to the future instrument but first creates a bar object based on the schema

        Args:
        schema: Schema.BAR - The schema of the bar
        roll_type: RollType - The roll type of the bar
        contract_type: ContractType - The contract type of the bar
        name: Optional[str] - The name of the bar

        Returns:
        None
        """
        contract: Contract = Contract(
            instrument=self.symbol,
            dataset=self.dataset,
            schema=schema,
        )

        if name is None:
            name = f"{contract.get_instrument()}-{roll_type}-{contract_type}"
        # Add a sleep to the task to avoid rate limiting
        await asyncio.sleep(3)
        try:
            await contract.construct_async(
                client=self.client, roll_type=roll_type, contract_type=contract_type
            )
            if contract_type == ContractType.FRONT:
                self.front = contract
                self.price = contract.backadjusted
            elif contract_type == ContractType.BACK:
                self.back = contract
            self._process_and_store_data(
                contract, schema, roll_type, contract_type
            )
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")

    @property
    def percent_returns(self) -> pd.Series:
        """
        Returns the percent returns of the future instrument

        Args:
        None

        Returns:
        pd.Series: The percent returns of the future instrument
        """

        if not hasattr(self, "_percent_change"):
            # * For equation see:
            # https://qoppac.blogspot.com/2023/02/percentage-or-price-differences-when.html
            self._percent_change: pd.Series = (
                self.price - self.price.shift(1)
            ) / self.front.get_close().shift(1)

            self._percent_change.name = self.name

        return self._percent_change    

def initialize_instruments(instrument_df : pd.DataFrame) -> list[Instrument]:
    return [
        Instrument(
            symbol=row.loc['dataSymbol'],
            dataset=DATASET.from_str(row.loc['dataSet']),
            currency=row.loc['currency'],
            exchange=row.loc['exchange'],
            security_type=SecurityType.from_str(row.loc['instrumentType']),
            multiplier=row.loc['multiplier'],
            ib_symbol=row.loc['ibSymbol']
        )
        for n, row in instrument_df.iterrows()
    ]

async def fetch_futures_data(futures : list[Future], rate: int = 5) -> None:
    """
    Fetches the data for the futures instruments asynchronously

    The fetch_futures_data function fetches the data for the futures instruments asynchronously using asyncio and a semaphore to limit the number of concurrent requests.

    Args:
    futures: list[Future] - The list of future instruments to fetch data for
    rate: int - The rate limit for the number of concurrent requests

    Returns:
    None
    """
    semaphore = asyncio.Semaphore(rate)
    async def fetch_with_semaphore(future: Future):
        async with semaphore:
            await future.add_data_async(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT)
    tasks = []
    for future in futures:
        task = asyncio.create_task(fetch_with_semaphore(future))
        tasks.append(task)

    await asyncio.gather(*tasks)

async def main():
    ex: str = "CME"
    bucket: list[str] = ["ES", "NQ", "RTY", "YM", "ZN"]
    multipliers: dict[str, float] = {
        "ES": 50,
        "NQ": 20,
        "RTY": 50,
        "YM": 5,
        "ZN": 1000,
    }
    futures: list[Future] = []

    tasks = []

    for sym in bucket:
        fut: Future = Future(symbol=sym, dataset=ex, multiplier=multipliers[sym])
        task = asyncio.create_task(fut.add_data_async(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT))
        tasks.append(task)
        futures.append(fut)

    await asyncio.gather(*tasks)

    print("Futures:")
    
    for fut in futures:
        print(fut.price)

if __name__ == "__main__":
    asyncio.run(main())