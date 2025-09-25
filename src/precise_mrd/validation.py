"""
Data validation module for pipeline boundary checks.

This module provides:
- Schema validation for pipeline outputs
- Data integrity checks
- Fail-fast validation at stage boundaries
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from .exceptions import PreciseMRDError


class ValidationError(PreciseMRDError):
    """Raised when data validation fails."""
    pass


class DataValidator:
    """Validate data at pipeline boundaries."""
    
    def __init__(self, schemas_dir: Path = None):
        """Initialize validator with schemas directory."""
        if schemas_dir is None:
            schemas_dir = Path(__file__).parent.parent.parent / "schemas"
        self.schemas_dir = schemas_dir
        self._schemas = {}
    
    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load validation schema from YAML file."""
        if schema_name not in self._schemas:
            schema_path = self.schemas_dir / f"{schema_name}.yml"
            if not schema_path.exists():
                raise ValidationError(f"Schema file not found: {schema_path}")
            
            with open(schema_path, 'r') as f:
                self._schemas[schema_name] = yaml.safe_load(f)
        
        return self._schemas[schema_name]
    
    def validate_umi_consensus(self, df: pd.DataFrame) -> None:
        """Validate UMI consensus output data."""
        schema = self.load_schema("umi_consensus")
        
        # Check required columns
        required_cols = schema.get("required_columns", [])
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValidationError(
                f"Missing required columns: {missing_cols}",
                details={"missing_columns": list(missing_cols)}
            )
        
        # Check data types
        column_types = schema.get("column_types", {})
        for col, expected_type in column_types.items():
            if col not in df.columns:
                continue
                
            if expected_type == "integer":
                if not pd.api.types.is_integer_dtype(df[col]):
                    raise ValidationError(f"Column '{col}' must be integer type")
            elif expected_type == "float":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValidationError(f"Column '{col}' must be numeric type")
            elif expected_type == "string":
                if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                    raise ValidationError(f"Column '{col}' must be string type")
        
        # Check constraints
        constraints = schema.get("constraints", {})
        for col, constraint_dict in constraints.items():
            if col not in df.columns:
                continue
                
            col_data = df[col]
            
            # Min/max constraints
            if "min" in constraint_dict:
                min_val = constraint_dict["min"]
                if col_data.min() < min_val:
                    raise ValidationError(
                        f"Column '{col}' has values below minimum ({min_val})",
                        details={"min_value": float(col_data.min()), "required_min": min_val}
                    )
            
            if "max" in constraint_dict:
                max_val = constraint_dict["max"]
                if col_data.max() > max_val:
                    raise ValidationError(
                        f"Column '{col}' has values above maximum ({max_val})",
                        details={"max_value": float(col_data.max()), "required_max": max_val}
                    )
            
            # Allowed values constraints
            if "allowed_values" in constraint_dict:
                allowed = set(constraint_dict["allowed_values"])
                actual_values = set(col_data.dropna().unique())
                invalid_values = actual_values - allowed
                if invalid_values:
                    raise ValidationError(
                        f"Column '{col}' contains invalid values: {invalid_values}",
                        details={"invalid_values": list(invalid_values), "allowed_values": list(allowed)}
                    )
        
        # Custom validation rules
        rules = schema.get("validation_rules", [])
        for rule in rules:
            rule_name = rule.get("name", "unnamed_rule")
            check_expr = rule.get("check", "")
            
            try:
                # Simple validation - extend as needed
                if "family_size > 0" in check_expr:
                    invalid_mask = df["family_size"] <= 0
                    if invalid_mask.any():
                        raise ValidationError(
                            f"Validation rule '{rule_name}' failed: {invalid_mask.sum()} rows have family_size <= 0"
                        )
                
                elif "quality >= 0 and quality <= 100" in check_expr:
                    invalid_mask = (df["quality"] < 0) | (df["quality"] > 100)
                    if invalid_mask.any():
                        raise ValidationError(
                            f"Validation rule '{rule_name}' failed: {invalid_mask.sum()} rows have invalid quality scores"
                        )
                        
            except Exception as e:
                raise ValidationError(f"Error evaluating validation rule '{rule_name}': {str(e)}")
    
    def validate_dataframe_basic(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None,
        min_rows: int = 1,
        name: str = "dataset"
    ) -> None:
        """Basic dataframe validation."""
        
        # Check if empty
        if df.empty:
            raise ValidationError(f"{name} is empty")
        
        # Check minimum rows
        if len(df) < min_rows:
            raise ValidationError(
                f"{name} has insufficient rows: {len(df)} < {min_rows}",
                details={"actual_rows": len(df), "required_min": min_rows}
            )
        
        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValidationError(
                    f"{name} missing required columns: {missing}",
                    details={"missing_columns": list(missing)}
                )
        
        # Check for all-null columns
        null_cols = []
        for col in df.columns:
            if df[col].isnull().all():
                null_cols.append(col)
        
        if null_cols:
            raise ValidationError(
                f"{name} has columns with all null values: {null_cols}",
                details={"null_columns": null_cols}
            )
    
    def validate_simulation_results(self, df: pd.DataFrame) -> None:
        """Validate simulation results data."""
        required_columns = [
            "allele_fraction", "umi_depth", "detected", "n_alt_umi",
            "total_umi", "pvalue", "run_id"
        ]
        
        self.validate_dataframe_basic(df, required_columns, name="simulation_results")
        
        # Specific validation for simulation results
        if df["allele_fraction"].min() < 0 or df["allele_fraction"].max() > 1:
            raise ValidationError("allele_fraction values must be between 0 and 1")
        
        if df["umi_depth"].min() <= 0:
            raise ValidationError("umi_depth values must be positive")
        
        if not df["detected"].isin([0, 1]).all():
            raise ValidationError("detected column must contain only 0 and 1")
        
        if df["pvalue"].min() < 0 or df["pvalue"].max() > 1:
            raise ValidationError("pvalue values must be between 0 and 1")


# Convenience functions for common validation patterns
def validate_umi_output(df: pd.DataFrame) -> None:
    """Validate UMI consensus output."""
    validator = DataValidator()
    validator.validate_umi_consensus(df)


def validate_simulation_output(df: pd.DataFrame) -> None:
    """Validate simulation results output."""
    validator = DataValidator()
    validator.validate_simulation_results(df)