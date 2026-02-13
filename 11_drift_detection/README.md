# Topic 11: Drift Detection

## What You'll Learn

This topic teaches you how to:
- Detect data drift (input distribution changes)
- Detect concept drift (model performance degrades)
- Set up monitoring with Evidently AI
- Create alerts for anomalies
- Analyze drift patterns
- Take corrective action

## Why Drift Detection?

### The Problem
Models degrade over time because:
- **Data drift**: Input data distribution changes
- **Concept drift**: Relationship between input/output changes
- **Model decay**: Model becomes outdated

### Impact
- **Reduced accuracy**: Model performs worse
- **Business impact**: Wrong predictions cost money
- **User experience**: Poor quality outputs
- **Compliance**: May violate regulations

## Types of Drift

### 1. Data Drift
Input data distribution changes.

**Example**: 
- Training: 80% English, 20% Spanish
- Production: 60% English, 40% Spanish

**Detection**: Compare input distributions.

### 2. Concept Drift
Relationship between input and output changes.

**Example**:
- Training: "hot" = positive sentiment
- Production: "hot" = negative sentiment (context changed)

**Detection**: Monitor prediction accuracy.

### 3. Prediction Drift
Model predictions distribution changes.

**Example**: Model starts predicting more positive labels.

**Detection**: Compare prediction distributions.

## Evidently AI

### What is Evidently?
Open-source tool for ML monitoring and drift detection.

### Features
- **Data drift detection**: Statistical tests
- **Model performance**: Accuracy monitoring
- **Data quality**: Missing values, outliers
- **Dashboards**: Visual reports

## Setup

### Installation
```bash
pip install evidently
```

### Basic Usage
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable

# Compare reference (training) vs current (production)
report = Report(metrics=[DataDriftTable()])
report.run(
    reference_data=train_data,
    current_data=production_data
)
report.show()
```

## Monitoring Pipeline

### 1. Collect Data
Store production inputs and predictions.

### 2. Compute Metrics
Calculate drift metrics periodically.

### 3. Compare to Baseline
Compare against training/reference data.

### 4. Alert on Drift
Send alerts when drift detected.

### 5. Take Action
Retrain model or investigate cause.

## Implementation

See `drift_detector.py` for complete implementation.

### Data Collection
```python
# Store production data
def log_prediction(input_data, prediction, model_version):
    store.append({
        "timestamp": datetime.now(),
        "input": input_data,
        "prediction": prediction,
        "model_version": model_version
    })
```

### Drift Detection
```python
from drift_detector import DriftDetector

# Initialize with reference data
detector = DriftDetector(reference_data=train_data)

# Detect drift
result = detector.detect_data_drift(current_data=production_data)

if result["drift_detected"]:
    send_alert("Data drift detected!")
```

## Metrics to Monitor

### Data Drift Metrics
- **PSI (Population Stability Index)**: Distribution similarity
- **Kolmogorov-Smirnov test**: Distribution differences
- **Chi-square test**: Categorical distribution

### Model Performance
- **Accuracy**: Overall correctness
- **Precision/Recall**: Per-class metrics
- **F1 score**: Balanced metric

### Prediction Drift
- **Prediction distribution**: How predictions change
- **Prediction by segment**: Per-group analysis

## Dashboards

### Evidently Dashboard
```python
from evidently.ui.dashboards import Dashboard

dashboard = Dashboard("Drift Monitoring")
dashboard.add_report(drift_report)
dashboard.show()
```

### Integration with Grafana
Export metrics to Prometheus, visualize in Grafana.

## Alerting Rules

### Data Drift Alert
```yaml
alert: DataDriftDetected
expr: drift_score > 0.2
for: 1h
annotations:
  summary: "Data drift detected"
```

### Performance Degradation
```yaml
alert: ModelPerformanceDegraded
expr: accuracy < 0.8
for: 2h
annotations:
  summary: "Model accuracy below threshold"
```

## Best Practices

1. **Establish baseline**: Use training data as reference
2. **Monitor continuously**: Check drift regularly
3. **Set thresholds**: Define what's "drift"
4. **Investigate causes**: Understand why drift occurs
5. **Document actions**: Track responses to drift
6. **Automate responses**: Auto-retrain or alert

## Common Scenarios

### Gradual Drift
Slow change over time → Retrain periodically

### Sudden Drift
Rapid change → Investigate cause immediately

### Seasonal Drift
Predictable patterns → Account for seasonality

## Exercises

1. **Set up Evidently**: Install and configure
2. **Detect data drift**: Compare training vs production
3. **Create dashboard**: Visualize drift metrics
4. **Set up alerts**: Alert on drift detection
5. **Investigate drift**: Analyze why drift occurred

## Next Steps

- **Topic 8**: Integrate drift detection with monitoring
- **Topic 10**: Track drift per model version
- **Topic 9**: Use drift detection in canary deployments

## Further Reading

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Drift Detection Methods](https://en.wikipedia.org/wiki/Concept_drift)
- [ML Monitoring Best Practices](https://www.evidentlyai.com/blog/ml-monitoring)

