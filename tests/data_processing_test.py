import pytest
from src.data_processing import DataProcessor
import src.polls as polls
import logging

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

def test_get_nat_polls():
    processor = DataProcessor()
    poll_avgs = processor.get_nat_polls()
    logging.warning(poll_avgs.shape)
    logging.warning(poll_avgs.columns)
    logging.warning(poll_avgs.index)
    assert len(poll_avgs.columns) == 5, "There should be 5 sets of poll averages."
    logging.warning(poll_avgs)
    assert all(
            party in poll_avgs.index for party in ["Con", "Lab", "Lib", "Nat", "Grn", "Ref", "Oth"]
        ), "Poll averages should contain all specified parties."

def test_get_features_and_labels():
    processor = DataProcessor()
    df_features, df_labels = processor.get_features_and_labels()
    assert not df_features.empty, "Features DataFrame should not be empty."
    assert not df_labels.empty, "Labels DataFrame should not be empty."
    assert df_features.shape[0] == df_labels.shape[0], "Features and labels should have the same number of rows."