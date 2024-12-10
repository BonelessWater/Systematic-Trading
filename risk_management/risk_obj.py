from abc import ABC, abstractmethod
from typing import Callable, Any, List, Dict
import numpy as np
import pandas as pd
from arch import arch_model

class Risk(ABC):
    """
    Abstract base class for risk management.
    """

    def __init__(self, data, capital):
        # Risk object methodology
        # self.constraints: List[Callable[[Dict[str, Any]], bool]] = []
        # self.metrics: List[Callable[[Dict[str, Any]], Any]] = []
        self.constraints: List[Callable[[Dict[str, Any], np.ndarray, float], bool]] = []
        self.metrics: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []

        self.logs: List[Dict[str, Any]] = []

        # All stock data
        self.data = data
        self.capital = capital

    @abstractmethod
    def base_constraints(self, data: Dict[str, Any], positions) -> bool:
        """
        Base constraints that all strategies must satisfy.
        """
        pass

    @abstractmethod
    def base_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base metrics that all strategies must log.
        """
        pass

    def add_constraint(self, constraint: Callable[[Dict[str, Any], Any], bool]) -> None:
        """
        Adds a constraint to the risk object

        Parameters
        ----------
        constraint : Callable[[Dict[str, Any]], bool]
            Function that takes in a dictionary and returns true or false whether the
            data meets the contraint.
        """
        self.constraints.append(constraint)
        # self.constraints.append(lambda data, positions: constraint(data, positions, **kwargs))

    def add_metric(self, metric: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Add a custom metric function.
        """
        self.metrics.append(metric)

    def check_constraints(self, data: Dict[str, Any], positions: np.ndarray) -> bool:
        """
        Check all constraints.
        """
        # Start with base constraints; temporarily commented
        '''
        if not self.base_constraints(data, positions):
            return False
        '''

        # Check custom constraints
        for constraint in self.constraints:
            
            if not constraint(data, positions, self.capital):
                print("custom")
                return False

        return True

    def evaluate_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate and log all metrics.
        """
        # Start with base metrics
        metrics = self.base_metrics(data)

        # Add custom metrics
        for metric in self.metrics:
            metrics.update(metric(data))

        # Log the metrics for reference
        self.logs.append(metrics)
        return metrics

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieve logs of all evaluated metrics.
        """
        return self.logs