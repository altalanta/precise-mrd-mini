
# MRD Pipeline Report

## Summary Metrics

- ROC AUC: 0.2181
- Average Precision: 0.5674
- Brier Score: 0.6435
- Detected Cases: 6 / 12

## Calibration

| Bin | Lower | Upper | Count | Event Rate | Confidence |
| --- | --- | --- | --- | --- | --- |

| 0 | 0.0 | 0.1 | 844 | 0.659 | 0.0 |

| 9 | 0.9 | 1.0 | 20 | 1.0 | 0.99 |


## Limit of Detection Grid

| Variant | Depth | Allele Fraction | Detection Rate |
| --- | --- | --- | --- |

| VAR01 | 1000 | 0.0 | 0.0 |

| VAR01 | 1000 | 0.005 | 0.035 |


## Raw Metrics JSON

```json
{
  "roc_auc": 0.21806278935185186,
  "roc_auc_ci": {
    "mean": 0.5173959679526267,
    "lower": 0.4751539730825155,
    "upper": 0.5550629825764841,
    "std": 0.021360191245833774
  },
  "average_precision": 0.5673776510226742,
  "average_precision_ci": {
    "mean": 0.6998969247095808,
    "lower": 0.6553719770623939,
    "upper": 0.7413075663108967,
    "std": 0.021013521276859536
  },
  "brier_score": 0.6435210764098248,
  "detected_cases": 6,
  "total_cases": 12,
  "calibration": [
    {
      "bin": 0,
      "lower": 0.0,
      "upper": 0.1,
      "count": 844,
      "event_rate": 0.6587677725118484,
      "confidence": 1.0000000000000003e-09
    },
    {
      "bin": 9,
      "lower": 0.9,
      "upper": 1.0,
      "count": 20,
      "event_rate": 1.0,
      "confidence": 0.9896117714538825
    }
  ]
}
```