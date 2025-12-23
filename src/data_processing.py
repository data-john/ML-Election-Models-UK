import pandas as pd
from src.polls import *


class DataProcessor:
    def __init__(self) -> None:
        self.prefer_percentages = True
        pass

    def load_constituency_data(self) -> pd.DataFrame:
        """Load constituency data from a CSV file."""
        df = pd.read_csv("data/resultsclusteredconstituencieslocationswinners.csv")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the constituency data."""
        unnamed_cols = [col for col in df.columns if "Unnamed:" in col]
        drop_cols = []
        for col in unnamed_cols:
            drop_cols.append(col)
        df = df.drop(columns=drop_cols)
        return df
    
    def get_raw_results_columns(self) -> list:
        """Get the list of raw results columns."""
        res_cols_05 = ['2005_Resultscon', '2005_Resultslab', '2005_Resultslib', '2005_Resultsref',
         '2005_Resultsnat', '2005_Resultsoth', '2005_Resultstot',]
        res_cols_10 = ['2010_Resultscon', '2010_Resultslab', '2010_Resultslib', '2010_Resultsref',
         '2010_Resultsnat', '2010_Resultsoth', '2010_Resultstot',]
        res_cols_15 = ['2015_Resultscon', '2015_Resultslab', '2015_Resultslib', '2015_Resultsref',
         '2015_Resultsnat', '2015_Resultsoth', '2015_Resultstot',]
        res_cols_17 = ['2017_Resultscon', '2017_Resultslab', '2017_Resultslib', '2017_Resultsref',
       '2017_Resultsnat', '2017_Resultsoth', '2017_Resultstot',]
        res_cols_19 = ['2019_Resultscon', '2019_Resultslab', '2019_Resultslib', '2019_Resultsref',
         '2019_Resultsnat', '2019_Resultsoth', '2019_Resultstot',]
        raw_results_cols = (res_cols_05 + res_cols_10 + res_cols_15 +
                            res_cols_17 + res_cols_19)
        return raw_results_cols

    def get_consituency_features(self):
        df = self.load_constituency_data()
        df = self.preprocess_data(df)
        drop_cols = []
        if self.prefer_percentages:
            raw_results_cols = self.get_raw_results_columns()
            drop_cols.extend(raw_results_cols)
        for col in ["ONS code", "New constituency name"]:
            drop_cols.append(col)
        df = df.drop(columns=drop_cols)
        return df
    
    def get_nat_polls(self) -> pd.DataFrame:
        poll_avgs = {}
        for year, url, col_dict in [
            ("2005",url_05, col_dict05),
            ("2010",url_10, col_dict10),
            ("2015",url_15, col_dict15),
            ("2017",url_17, col_dict17),
            ("2019",url_19, col_dict19),
            # ("2024",url_24, col_dict24),
        ]:
            poll_avgs[year] = get_weighted_poll_avg(url, col_dict)
        return pd.DataFrame(poll_avgs)

    def get_polling_features(self):
        return self.get_nat_polls() #TBC