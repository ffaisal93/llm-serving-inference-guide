"""
Drift Detection with Evidently AI

This module demonstrates how to detect data drift and concept drift
in LLM serving applications.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    PredictionDriftMetric,
    ColumnDriftMetric
)
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects drift in production data compared to reference (training) data.
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: Training/reference dataset (baseline)
        """
        self.reference_data = reference_data
        self.column_mapping = ColumnMapping()
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.2
    ) -> Dict:
        """
        Detect data drift in input features.
        
        Args:
            current_data: Current production data
            threshold: PSI threshold for drift (0.2 = 20% drift)
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Running data drift detection...")
        
        # Create report
        report = Report(metrics=[
            DataDriftTable(),
            DatasetDriftMetric()
        ])
        
        # Run comparison
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Get results
        dataset_metric = report.get_metric(DatasetDriftMetric())
        data_drift_table = report.get_metric(DataDriftTable())
        
        # Check if drift detected
        drift_detected = dataset_metric.dataset_drift
        drift_score = dataset_metric.number_of_drifted_columns / len(current_data.columns) if len(current_data.columns) > 0 else 0
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "threshold": threshold,
            "drifted_columns": [
                col for col, metric in data_drift_table.get_metric().items()
                if metric.drift_detected
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"Data drift detected! Score: {drift_score:.2f}")
        else:
            logger.info("No data drift detected")
        
        return result
    
    def detect_prediction_drift(
        self,
        current_data: pd.DataFrame,
        prediction_column: str = "prediction"
    ) -> Dict:
        """
        Detect drift in model predictions.
        
        Args:
            current_data: Current production data with predictions
            prediction_column: Name of prediction column
            
        Returns:
            Dictionary with prediction drift results
        """
        logger.info("Running prediction drift detection...")
        
        # Ensure reference has predictions
        if prediction_column not in self.reference_data.columns:
            raise ValueError(f"Prediction column '{prediction_column}' not in reference data")
        
        # Create report
        report = Report(metrics=[
            PredictionDriftMetric(),
            ColumnDriftMetric(column_name=prediction_column)
        ])
        
        # Run comparison
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Get results
        prediction_metric = report.get_metric(PredictionDriftMetric())
        column_metric = report.get_metric(ColumnDriftMetric(column_name=prediction_column))
        
        result = {
            "drift_detected": prediction_metric.drift_detected,
            "drift_score": column_metric.drift_score,
            "reference_mean": self.reference_data[prediction_column].mean(),
            "current_mean": current_data[prediction_column].mean(),
            "timestamp": datetime.now().isoformat()
        }
        
        if result["drift_detected"]:
            logger.warning(f"Prediction drift detected! Score: {result['drift_score']:.2f}")
        
        return result
    
    def generate_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Report:
        """
        Generate comprehensive drift report.
        
        Args:
            current_data: Current production data
            output_path: Optional path to save HTML report
            
        Returns:
            Evidently report object
        """
        logger.info("Generating drift report...")
        
        # Create comprehensive report
        report = Report(metrics=[
            DataDriftTable(),
            DatasetDriftMetric(),
            PredictionDriftMetric()
        ])
        
        # Run comparison
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report if path provided
        if output_path:
            report.save_html(output_path)
            logger.info(f"Report saved to {output_path}")
        
        return report


# Example usage
if __name__ == "__main__":
    # Load reference data (training data)
    reference_data = pd.DataFrame({
        "prompt_length": [10, 20, 30, 40, 50],
        "prediction": [0.5, 0.6, 0.7, 0.8, 0.9]
    })
    
    # Initialize detector
    detector = DriftDetector(reference_data)
    
    # Simulate current data (with drift)
    current_data = pd.DataFrame({
        "prompt_length": [100, 200, 300, 400, 500],  # Different distribution
        "prediction": [0.3, 0.4, 0.5, 0.6, 0.7]  # Different predictions
    })
    
    # Detect drift
    data_drift = detector.detect_data_drift(current_data)
    print("Data Drift:", data_drift)
    
    prediction_drift = detector.detect_prediction_drift(current_data)
    print("Prediction Drift:", prediction_drift)
    
    # Generate report
    report = detector.generate_report(current_data, "drift_report.html")
    print("Report generated!")

