
# MRD Pipeline Report

## Summary Metrics

- ROC AUC: 0.1723
- Average Precision: 0.5017
- Brier Score: 0.592
- Detected Cases: 26 / 30

## Calibration

| Bin | Lower | Upper | Count | Event Rate | Confidence |
| --- | --- | --- | --- | --- | --- |

| 0 | 0.0 | 0.1 | 4456 | 0.612 | 0.0 |

| 9 | 0.9 | 1.0 | 152 | 1.0 | 0.99 |


## Limit of Detection Grid

| Variant | Depth | Allele Fraction | Detection Rate |
| --- | --- | --- | --- |

| VAR01 | 1000 | 0.0 | 0.0 |

| VAR01 | 1000 | 0.01 | 0.094 |

| VAR01 | 2000 | 0.0 | 0.0 |

| VAR01 | 2000 | 0.01 | 0.11 |

| VAR02 | 1000 | 0.0 | 0.0 |

| VAR02 | 1000 | 0.005 | 0.052 |

| VAR02 | 2000 | 0.0 | 0.0 |

| VAR02 | 2000 | 0.005 | 0.04 |

| VAR03 | 1000 | 0.0 | 0.0 |

| VAR03 | 1000 | 0.001 | 0.013 |

| VAR03 | 2000 | 0.0 | 0.0 |

| VAR03 | 2000 | 0.001 | 0.008 |


## Raw Metrics JSON

```json
{
  "roc_auc": 0.17233555169753087,
  "roc_auc_ci": {
    "mean": 0.5261263524000725,
    "lower": 0.5130135900970787,
    "upper": 0.541442699535854,
    "std": 0.008307364621400689
  },
  "average_precision": 0.5016862785493542,
  "average_precision_ci": {
    "mean": 0.6738248274010108,
    "lower": 0.6573187178860339,
    "upper": 0.6908819913921858,
    "std": 0.0099443260002189
  },
  "brier_score": 0.5920174513662756,
  "detected_cases": 26,
  "total_cases": 30,
  "calibration": [
    {
      "bin": 0,
      "lower": 0.0,
      "upper": 0.1,
      "count": 4456,
      "event_rate": 0.6122082585278277,
      "confidence": 1e-09
    },
    {
      "bin": 9,
      "lower": 0.9,
      "upper": 1.0,
      "count": 152,
      "event_rate": 1.0,
      "confidence": 0.9900645347197844
    }
  ]
}
```