from abc import ABC, abstractmethod
from typing import Callable, Any, List, Dict

class Risk(ABC):
    """
    Abstract base class for risk management.
    """

    def __init__(self):
        self.constraints: List[Callable[[Dict[str, Any]], bool]] = []
        self.metrics: List[Callable[[Dict[str, Any]], Any]] = []
        self.logs: List[Dict[str, Any]] = []

    @abstractmethod
    def base_constraints(self, data: Dict[str, Any]) -> bool:
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

    def add_constraint(self, constraint: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Add a custom constraint function.
        """
        self.constraints.append(constraint)

    def add_metric(self, metric: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Add a custom metric function.
        """
        self.metrics.append(metric)

    def check_constraints(self, data: Dict[str, Any]) -> bool:
        """
        Check all constraints.
        """
        # Start with base constraints
        if not self.base_constraints(data):
            return False
        
        # Check custom constraints
        for constraint in self.constraints:
            if not constraint(data):
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
        
        self.logs.append(metrics)
        return metrics

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieve logs of all evaluated metrics.
        """
        return self.logs

# Example subclass implementation
class ExampleRisk(Risk):
    def base_constraints(self, data: Dict[str, Any]) -> bool:
        # Example: Check if capital is above a threshold
        return data.get("capital", 0) > 1000

    def base_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Example: Calculate a simple risk metric
        return {"leverage": data.get("capital", 1) / data.get("exposure", 1)}
