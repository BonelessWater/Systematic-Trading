import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Self, Optional, TypeVar, Generic

from risk_management.utility_functions.instrument import Instrument

class _utils:
    @staticmethod
    def ffill_zero(df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill zeros in a DataFrame. This function will replace all zeros with the last non-zero value in the DataFrame.
        """
        # We assume any gaps in percent returns at this point are because the market was closed that day,
        # but only fill forward;
        # Find the index of the first non-NaN value in each column
        first_non_nan_index = df.apply(lambda x: x.first_valid_index())

        # Iterate over each column and replace NaN values below the first non-NaN index with 0
        for column in df.columns:
            first_index = first_non_nan_index[column]
            if first_index is not None:
                # Extract the relevant part of the column as a Series
                series_to_fill = df.loc[first_index:, column]
                # Fill NaNs with 0
                filled_series = series_to_fill.fillna(0)
                # Reassign the filled series back to the DataFrame
                df.loc[first_index:, column] = filled_series.values

        return df

class StandardDeviation(pd.DataFrame):
    def __init__(self, data : pd.DataFrame = None) -> None:
        super().__init__(data)
        self.__is_annualized : bool = False

    def annualize(self, inplace=False) -> Optional[Self]:
        if self.__is_annualized:
            return self

        factor : float = 256 ** 0.5

        if inplace:
            self *= factor
            self.__is_annualized = True
            return None

        new = StandardDeviation(self)
        new.annualize(inplace=True)
        return new

    def to_variance(self) -> 'Variance':
        return Variance(self ** 2)
    
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self)

class Variance(pd.DataFrame):
    def __init__(self, data : pd.DataFrame = None) -> None:
        super().__init__(data)
        self.__is_annualized = False

    def annualize(self, inplace=False) -> Optional[Self]:
        if self.__is_annualized:
            return self

        factor : float = 256

        if inplace:
            self *= factor
            self.__is_annualized = True
            return None

        new = Variance(self)
        new = new.annualize(inplace=True)
        return new
    
    def to_standard_deviation(self) -> 'StandardDeviation':
        return StandardDeviation(self ** 0.5)
    
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self)

class Covariance:
    def __init__(self, covariance_matrices : np.ndarray = None, dates : pd.DatetimeIndex = None, instrument_names : list[str] = None) -> None:
        self._covariance_matrices = covariance_matrices
        self._dates = dates
        self._instrument_names = instrument_names if instrument_names is not None else []
        self._columns = []

    def to_frame(self) -> pd.DataFrame:
        self._columns = [f'{instrument_I}_{instrument_J}' for i, instrument_I in enumerate(self._instrument_names) for j, instrument_J in enumerate(self._instrument_names) if i <= j]
        rows = []
        for n in range(self._covariance_matrices.shape[0]):
            row = []
            for i, instrument_I in enumerate(self._instrument_names):
                for j, instrument_J in enumerate(self._instrument_names):
                    if i > j:
                        continue
                    row.append(self._covariance_matrices[n, i, j])
            rows.append(row)
        return pd.DataFrame(rows, columns=self._columns, index=self._dates)

    # @property
    # def columns(self) -> list[str]:
    #     return self._columns
    
    @property
    def index(self) -> pd.DatetimeIndex:
        return self._dates
    
    def reindex(self, dates : pd.DatetimeIndex):
        intersection = self._dates.intersection(dates)

        self._covariance_matrices = self._covariance_matrices[self._dates.isin(intersection)]
        self._dates = intersection

    def from_frame(self, df : pd.DataFrame, inplace : bool = False) -> Optional['Covariance']:
        for column in df.columns:
            if column.split('_')[0] not in self._instrument_names:
                self._instrument_names.append(column.split('_')[0])
        
        self._dates = pd.to_datetime(df.index)

        self._covariance_matrices = np.zeros((df.shape[0], len(self._instrument_names), len(self._instrument_names)))
        for n, (index, row) in enumerate(df.iterrows()):
            for i, instrument_I in enumerate(self._instrument_names):
                for j, instrument_J in enumerate(self._instrument_names):
                    if i > j:
                        continue
                    self._covariance_matrices[n, i, j] = row[f'{instrument_I}_{instrument_J}']
                    self._covariance_matrices[n, j, i] = row[f'{instrument_I}_{instrument_J}']
        
        if not inplace:
            return self

    def iterate(self):
        for n in range(self._covariance_matrices.shape[0]):
            yield self._dates[n], self._covariance_matrices[n]

    def dropna(self):
        valid_indices = ~np.isnan(self._covariance_matrices).any(axis=(1, 2))
        self._covariance_matrices = self._covariance_matrices[valid_indices]
        self._dates = self._dates[valid_indices]

    @property
    def empty(self) -> bool:
        return self._covariance_matrices is None
    
    @property
    def iloc(self):
        return self._ILocIndexer(self)

    @property
    def loc(self):
        return self._LocIndexer(self)

    def __str__(self) -> str:
        return self.to_frame().__str__()
    
    def __repr__(self) -> str:
        return self.to_frame().__repr__()
    
    def __getitem__(self, key):
        return self._covariance_matrices[key]

    class _ILocIndexer:
        def __init__(self, parent : 'Covariance') -> None:
            self.parent = parent

        def __getitem__(self, key):
            covariance_matrix = self.parent._covariance_matrices[key]
            return pd.DataFrame(covariance_matrix, index=self.parent._instrument_names, columns=self.parent._instrument_names)

    class _LocIndexer:
        def __init__(self, parent : 'Covariance') -> None:
            self.parent = parent

        def __getitem__(self, key):
            covariance_matrix = self.parent._covariance_matrices[self.parent._dates.get_loc(key)]
            return pd.DataFrame(covariance_matrix, index=self.parent._instrument_names, columns=self.parent._instrument_names)
        

T = TypeVar('T', bound=Instrument)

class RiskMeasure(ABC, Generic[T]):
    def __init__(self, tau : float = None) -> None:
        self.instruments : list[Instrument]

        self.__returns = pd.DataFrame()
        self.__product_returns : pd.DataFrame = pd.DataFrame()
        self.__jump_covariances : Covariance = Covariance()
        self.fill : bool

        if tau is not None:
            self.tau = tau

    @property
    def tau(self) -> float:
        if not hasattr(self, '_tau'):
            raise ValueError("tau is not set")
        return self._tau
    
    @tau.setter
    def tau(self, value : float) -> None:
        if (value < 0) or not isinstance(value, float):
            raise ValueError("tau, x, is a float such that x ∈ (0, inf)")
        self._tau = value

    def get_returns(self) -> pd.DataFrame:
        if not self.__returns.empty:
            return self.__returns

        returns = pd.DataFrame()
        for instrument in self.instruments:
            returns = pd.concat([returns, instrument.percent_returns], axis=1)

        returns = returns.reindex(sorted(returns.columns), axis=1)

        if self.fill:
            returns = _utils.ffill_zero(returns)

        return returns

    def get_product_returns(self) -> pd.DataFrame:
        if not self.__product_returns.empty:
            return self.__product_returns

        returns = self.get_returns()

        product_dictionary : dict[str, pd.Series] = {}

        for i, instrument_i in enumerate(self.instruments):
            for j, instrument_j in enumerate(self.instruments):
                if i > j:
                    continue
                
                product_dictionary[f'{instrument_i.name}_{instrument_j.name}'] = returns[instrument_i.name] * returns[instrument_j.name]

        self.__product_returns = pd.DataFrame(product_dictionary, index=returns.index)

        self.__product_returns = _utils.ffill_zero(self.__product_returns) if self.fill else self.__product_returns

        return self.__product_returns

    @abstractmethod
    def get_var(self) -> Variance:
        pass

    @abstractmethod
    def get_cov(self) -> Covariance:
        pass
    
    def get_jump_cov(self, percentile : float, window : int) -> Covariance:
        if not self.__jump_covariances.empty:
            return self.__jump_covariances
    
        if (percentile < 0) or (percentile > 1):
            raise ValueError("percentile, x, is a float such that x ∈ (0, 1)")

        covar_df = self.get_cov().to_frame()

        dates = covar_df.index

        jump_covariances = pd.DataFrame(index=dates, columns=covar_df.columns, dtype=np.float64)

        for i in range(len(dates)):
            if i < window:
                continue

            window_covariances = covar_df.iloc[i-window:i]
            jump_covariances.iloc[i] = window_covariances.quantile(percentile)

        jump_covariances = jump_covariances.interpolate().bfill() if self.fill else jump_covariances

        self.__jump_covariances = Covariance().from_frame(jump_covariances)

        return self.__jump_covariances

class CRV(RiskMeasure[T]):
    def __init__(self, risk_target : float, instruments : list[T], window : int, span : int, fill : bool = True) -> None:
        super().__init__(tau=risk_target)

        self.instruments = instruments
        self.fill = fill
        self.window = window
        self.span = span

        self.__var = Variance()
        self.__cov = Covariance()
        self.__weekly_returns = pd.DataFrame()

    def get_var(self) -> Variance:
        if not self.__var.empty:
            return self.__var

        returns = self.get_returns()

        daily_exp_std = returns.ewm(span=self.span).std()
        annualized_exp_std : pd.DataFrame = daily_exp_std * (256 ** 0.5)

        ten_year_vol = annualized_exp_std.rolling(
            window=256 * 10, min_periods=1
        ).mean()
        weighted_vol = 0.3 * ten_year_vol + 0.7 * annualized_exp_std
        weighted_vol.dropna(inplace=True)
        weighted_vol = weighted_vol.interpolate() if self.fill else weighted_vol

        self.__var = Variance(weighted_vol / (256 ** 0.5) ** 2)

        return self.__var
    
    def get_weekly_returns(self) -> pd.DataFrame:
        if not self.__weekly_returns.empty:
            return self.__weekly_returns

        # add 1 to each return to get the multiplier
        return_multipliers = self.get_returns() + 1

        # ! there is some statistical error here because the first weekly return could be
        # ! less than 5 returns but this should be insignificant for the quantity of returns
        n = len(return_multipliers)
        complete_groups_index = n // 5 * 5  # This will give the largest number less than n that is a multiple of 5
        sliced_return_multipliers = return_multipliers[:complete_groups_index]

        # group into chunks of 5, and then calculate the product of each chunk, - 1 to get the return
        weekly_returns = sliced_return_multipliers.groupby(np.arange(complete_groups_index) // 5).prod() - 1

        # Use the last date in each chunk
        weekly_returns.index = self.get_returns().index[4::5]

        nan_mask = self.get_returns().isna().copy()

        weekly_returns = weekly_returns.where(~nan_mask.iloc[4::5])

        self.__weekly_returns = weekly_returns.interpolate() if self.fill else weekly_returns

        return self.__weekly_returns

    def get_corr(self) -> pd.DataFrame:
        return self.get_weekly_returns().rolling(window=self.window).corr()

    def get_cov(self) -> Covariance:
        if not self.__cov.empty:
            return self.__cov

        rolling_corr : pd.DataFrame = self.get_corr()
        vol : pd.DataFrame = self.get_var().to_standard_deviation().annualize().to_frame()

        # weekly_dates = rolling_corr.index.levels[0]
        weekly_dates = rolling_corr.index.get_level_values(0).unique()

        vol : pd.DataFrame = vol.reindex(weekly_dates)

        if len(vol) != len(weekly_dates):
            raise ValueError("vol and dates do not match")

        covs = []

        all_dates = self.get_returns().index

        last_cov = np.zeros((len(vol.columns), len(vol.columns)))
        last_cov[last_cov == 0] = np.nan

        for date in all_dates:
            if date in weekly_dates:
                vol_matrix : pd.Series = vol.loc[date]
                corr_matrix = rolling_corr.xs(date, level=0)
                cov = np.diag(vol_matrix.values) @ corr_matrix.values @ np.diag(vol_matrix.values)
                last_cov = cov            
            covs.append(last_cov)

        i : int = 0
        while np.isnan(covs[i]).any():
            i += 1

        covs = covs[i:]
        all_dates = all_dates[i:]

        self.__cov = Covariance(np.array(covs), all_dates, vol.columns)

        return self.__cov


class EqualWeight(RiskMeasure[T]):
    def __init__(
        self,
        risk_target : float,
        instruments : list[T],
        window : int,
        fill : bool = True) -> None:
        
        super().__init__(tau=risk_target)

        self.instruments = instruments
        self.fill = fill
        self.window = window

        self.__var = Variance()
        self.__cov = Covariance()
    
    def get_var(self) -> Variance:
        if not self.__var.empty:
            return self.__var

        covar : Covariance = self.get_cov()
        variances = []
        for (date, matrix) in covar.iterate():
            variances.append(np.diag(matrix))

        self.__var = Variance(pd.DataFrame(variances, index=covar._dates, columns=covar._instruments))

        return self.__var

    def get_cov(self) -> Covariance:
        if self.__cov.empty:
            returns_matrix = self.get_returns().values

            covariance_matrices = np.zeros((returns_matrix.shape[0], returns_matrix.shape[1], returns_matrix.shape[1]))

            for i in range(1, returns_matrix.shape[0]):
                returns = returns_matrix[i-self.window:i+1]
                covariance_matrix = returns.T @ returns / self.window
                covariance_matrices[i] = covariance_matrix

            self.__cov = Covariance(covariance_matrices, self.get_returns().index, self.get_returns().columns)

        return self.__cov
    
    def get_corr(self) -> pd.DataFrame:
        # Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
        # corr = Dinv @ cov @ Dinv
        pass

class Simple(RiskMeasure[T]):
    def __init__(
        self,
        risk_target : float,
        instruments : list[T],
        window : int,
        fill : bool = True) -> None:
        
        super().__init__(tau=risk_target)

        self.instruments = instruments
        self.fill = fill
        self.window = window

        self.__var = Variance()
        self.__cov = Covariance()
    
    def get_var(self) -> Variance:
        if not self.__var.empty:
            return self.__var

        returns = self.get_returns()

        variance = returns.var()

        self.__var = Variance(variance)

        return self.__var
    
    def get_cov(self) -> Covariance:
        if self.__cov.empty:
            covar = pd.DataFrame(index=self.get_product_returns().index, columns=self.get_product_returns().columns, dtype=float)

            covar = self.get_product_returns().rolling(window=self.window).mean().bfill()

            covar = covar.interpolate() if self.fill else self.__cov
            covar = covar.iloc[self.window:, :]
            self.__cov = Covariance().from_frame(covar)

        return self.__cov

class GARCH(RiskMeasure[T]):
    def __init__(
        self,
        risk_target : float,
        instruments : list[T],
        weights : tuple[float, float, float],
        minimum_observations : int,
        fill : bool = True) -> None:

        super().__init__(tau=risk_target)

        self.instruments = instruments
        self.weights = weights
        self.minimum_observations = minimum_observations
        self.fill = fill

        self.__var = Variance()
        self.__cov = pd.DataFrame()

    def get_var(self) -> Variance:
        if not self.__var.empty:
            return self.__var
        
        if not self.__cov.empty:
            for name in self.__cov.columns:
                if '_' not in name:
                    continue
                if name.split('_')[0] != name.split('_')[1]:
                    continue
                self.__var[name.split('_')[0]] = self.__cov[name]
            return self.__var
        
        variance : pd.DataFrame = pd.DataFrame()

        for i, instrument in enumerate(self.get_returns().columns.tolist()):
            squared_returns = self.get_returns()[instrument] ** 2
            squared_returns.dropna(inplace=True)

            dates = squared_returns.index

            # Calculate rolling LT variance
            LT_variances = squared_returns.rolling(window=self.minimum_observations).mean().bfill()

            df = pd.Series(index=dates)
            df.iloc[0] = squared_returns.iloc[0]

            for j, _ in enumerate(dates[1:], 1):
                df.iloc[j] = squared_returns.iloc[j] * self.weights[0] + df.iloc[j-1] * self.weights[1] + LT_variances.iloc[j] * self.weights[2]

            if i == 0:
                variance = df.to_frame(instrument)
                continue

            variance = pd.merge(variance, df.to_frame(instrument), how='outer', left_index=True, right_index=True)

        variance = variance.interpolate() if self.fill else variance

        self.__var : Variance = Variance(variance[self.minimum_observations:])

        return self.__var

    def get_cov(self) -> Covariance:
        if not self.__cov.empty:
            return self.__cov

        product_returns : np.ndarray = self.get_product_returns().dropna().values
        LT_covariances : np.ndarray = self.get_product_returns().rolling(window=self.minimum_observations).mean().bfill().values

        covar = pd.DataFrame(index=self.get_product_returns().index, columns=self.get_product_returns().columns, dtype=float)
        covar.iloc[0] = product_returns[0]

        for i in range(1, len(product_returns)):
            covar.iloc[i] = product_returns[i] * self.weights[0] + covar.iloc[i-1] * self.weights[1] + LT_covariances[i] * self.weights[2]

        covar = covar.interpolate() if self.fill else covar
        covar = covar.iloc[self.minimum_observations:, :]

        self.__cov = Covariance().from_frame(covar)

        return self.__cov

    def get_jump_cov(self, percentile : float, window : int) -> pd.DataFrame:
        return super().get_jump_cov(percentile=percentile, window=window)
