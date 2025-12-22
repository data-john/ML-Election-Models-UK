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
    assert len(poll_avgs) == 6, "There should be 6 sets of poll averages."
    logging.warning(poll_avgs)
    # for poll_avg in poll_avgs: ### Consider updating nat_polls output to have consistent party names
    #     assert all(
    #         party in poll_avg.columns for party in ["Con", "Lab", "Lib Dem", "SNP", "Green", "UKIP", "Others"]
    #     ), "Poll averages should contain all specified parties."