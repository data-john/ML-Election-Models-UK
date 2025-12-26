from src.data_processing import DataProcessor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

class Preprocessor:
    def __init__(self) -> None:
        self.data_processor = DataProcessor()
    
    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the features DataFrame."""

        features.fillna(0, inplace=True)
        # Example preprocessing: Scaling numerical features and encoding categorical features
        numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = features.select_dtypes(include=['object']).columns

        scaler = MinMaxScaler()
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(features[categorical_cols])
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

        features = features.drop(columns=categorical_cols).reset_index(drop=True)
        features = pd.concat([features, encoded_cat_df], axis=1)

        return features
    

class ModelsEngine:
    def __init__(self) -> None:
        self.data_processor = DataProcessor()
        self.preprocessor = Preprocessor()
    
    def run_modeling_pipeline(self):
        features, labels = self.data_processor.load_features_and_labels()
        processed_features = self.preprocessor.preprocess_features(features)
        print("Processed Features Shape:", processed_features.shape)
        print(processed_features.info())
        model = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Model R^2 Score: {score}")

        
        

if __name__ == "__main__":
    engine = ModelsEngine()
    engine.run_modeling_pipeline()