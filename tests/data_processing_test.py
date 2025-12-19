import pytest
from src.data_processing import DataProcessor

def test_load_constituency_data():
    processor = DataProcessor()
    df = processor.load_constituency_data()
    assert not df.empty, "DataFrame should not be empty after loading data."

def test_get_consituency_features():
    processor = DataProcessor()
    df = processor.get_consituency_features()
    assert "ONS code" not in df.columns, "'ONS code' column should be dropped."
    assert "New constituency name" not in df.columns, "'New constituency name' column should be dropped."
    assert len(df.columns) ==54, "DataFrame should have 54 columns after dropping specified columns."