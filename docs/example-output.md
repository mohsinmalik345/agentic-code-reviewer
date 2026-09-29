# Example deployment output

This illustrative response shows the output shape. IDs and paths are examples, not claims about the Beelinks repositories.

```json
{
  "report": {
    "id": "deployment-report:8d1537c2c3dc894bf2d61eaa",
    "recommendation": "BLOCK",
    "risk_score": 70,
    "implementation_satisfies_intent": true,
    "breakage_risk": "high",
    "hidden_side_effects": [
      "The retry path can duplicate a downstream notification when the first request succeeds but its response is lost."
    ],
    "regression_test_node_ids": [
      "node:5f58a39bf4dd767892027790"
    ],
    "reasoning": [
      "A database write and an outbound API call are reachable from the changed function.",
      "Deterministic high signal database-write-impact established a risk floor of 70."
    ],
    "change": {
      "repository_id": "email-service",
      "changed_files": [
        {
          "path": "src/services/delivery.ts",
          "status": "modified",
          "additions": 18,
          "deletions": 4
        }
      ]
    },
    "impact": {
      "affected_services": [
        {
          "id": "node:2e97254853907d2d70625bf7",
          "name": "Beelinks Email Service"
        }
      ],
      "affected_endpoints": [],
      "potential_regressions": [
        "Regression-test database write success, validation, and failure paths."
      ]
    }
  },
  "markdown_report_path": "artifacts/reports/deployment-report-....md",
  "json_report_path": "artifacts/reports/deployment-report-....json"
}
```

A report is blocked when the configured deployment-risk model recommends blocking, intent is not satisfied, traversal is truncated, a critical signal exists, or the final risk score is 70 or higher. Deterministic high/critical signals establish score floors that the model cannot reduce.

