import pandas as pd
from src.polls import *
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
        res_cols_05 = ['2005_Results','2005_Resultscon', '2005_Resultslab', '2005_Resultslib', '2005_Resultsref',
         '2005_Resultsnat', '2005_Resultsoth', '2005_Resultstot',]
        res_cols_10 = ['2010_Results','2010_Resultscon', '2010_Resultslab', '2010_Resultslib', '2010_Resultsref',
         '2010_Resultsnat', '2010_Resultsoth', '2010_Resultstot',]
        res_cols_15 = ['2015_Results','2015_Resultscon', '2015_Resultslab', '2015_Resultslib', '2015_Resultsref',
         '2015_Resultsnat', '2015_Resultsoth', '2015_Resultstot',]
        res_cols_17 = ['2017_Results','2017_Resultscon', '2017_Resultslab', '2017_Resultslib', '2017_Resultsref',
       '2017_Resultsnat', '2017_Resultsoth', '2017_Resultstot',]
        res_cols_19 = ['2019_Results','2019_Resultscon', '2019_Resultslab', '2019_Resultslib', '2019_Resultsref',
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
        '''
        Year polling average for each party (To be expanded to include granular polling by demographic)

        '''

        return self.get_nat_polls() #TBC
    
    def get_demographic_columns(self, constituency_df: pd.DataFrame) -> list:
        demographic_cols = [col for col in constituency_df.columns if "Aged" in col]
        other_demo_cols = [
            "Region",
            "Total population",
            "Latitude",
            "Longitude",]
        return demographic_cols + other_demo_cols

    
    def get_features_and_labels(self):
        '''
        The features should be year agnostic:
        - That year polling average for each party (To be expanded to include granular polling by demographic)
        - Demographic features from the constituency data (Is this updated/year specific?)
        - Previous constituency election results (percentages)
        '''
        constituency_features = self.get_consituency_features()
        polling_features = self.get_polling_features()
        # Per Year
        election_years = ["2005", "2010", "2015", "2017", "2019"]
        def get_prev_election_year(year):
            year_idx = election_years.index(year)
            if year_idx == 0:
                return None  # No previous election
            return election_years[year_idx - 1]
        features_per_year = {}
        labels_per_year = {}
        results_parties = ["con", "lab", "lib", "nat", "ref", "oth"]
        for year in ["2010", "2015", "2017", "2019"]:
            # Labels are the constituency results for that year
            label_cols = [col for col in constituency_features.columns if col.startswith(f"{year}_Results")]
            labels_dict = {}
            for party in results_parties:
                for col in label_cols:
                    if party in col:
                        labels_dict[party] = col

                
            labels = constituency_features[labels_dict.values()].copy()
            labels.rename(columns={v: k for k, v in labels_dict.items()}, inplace=True)
            # Features include polling features for that year and constituency features
            poll_feats = polling_features[year]
            transposed_poll_feats = pd.DataFrame(poll_feats).T.reset_index(drop=True)
            # Fill down to match constituency rows
            transposed_poll_feats = pd.concat([transposed_poll_feats]*constituency_features.shape[0], ignore_index=True) 
            prev_result_cols = [col for col in constituency_features.columns if col.startswith(f"{get_prev_election_year(year)}_Results")]
            prev_results_feats = constituency_features[prev_result_cols]
            results_rename_dict = {}
            for party in results_parties:
                for col in prev_result_cols:
                    if party in col:
                        results_rename_dict[col] = f"Prev_{party}" 
            logging.warning(results_rename_dict)
            prev_results_feats.rename(columns=results_rename_dict, inplace=True)
            demographic_feats = constituency_features[self.get_demographic_columns(constituency_features)]
            yr_features = pd.concat([transposed_poll_feats, prev_results_feats, demographic_feats], axis=1)
            yr_labels = labels
            features_per_year[year] = yr_features
            labels_per_year[year] = yr_labels
        # combine all years
        all_features = pd.concat(features_per_year.values(), ignore_index=True)
        all_labels = pd.concat(labels_per_year.values(), ignore_index=True)
        return all_features, all_labels
    
    def save_features_and_labels(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:
        feature_path = "data/processed/features.csv"
        label_path = "data/processed/labels.csv"
        
        features.to_csv(feature_path, index=False)
        labels.to_csv(label_path, index=False)

    def load_features_and_labels(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_path = "data/processed/features.csv"
        label_path = "data/processed/labels.csv"
        
        features = pd.read_csv(feature_path)
        labels = pd.read_csv(label_path)
        return features, labels
    



if __name__ == "__main__":
    processor = DataProcessor()
    features, labels = processor.get_features_and_labels()
    processor.save_features_and_labels(features, labels)