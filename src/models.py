from src.data_processing import DataProcessor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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
    
    def preprocess_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the labels DataFrame."""
        scaler = MinMaxScaler()
        labels = pd.DataFrame(scaler.fit_transform(labels), columns=labels.columns)
        return labels

class ModelsEngine:
    def __init__(self) -> None:
        self.data_processor = DataProcessor()
        self.preprocessor = Preprocessor()
        self.model = LinearRegression()
    
    def run_modeling_pipeline(self):
        features, labels = self.data_processor.load_features_and_labels()
        processed_features = self.preprocessor.preprocess_features(features)
        print("Processed Features Shape:", processed_features.shape)
        # print(processed_features.info())
        model = self.model
        X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Model R^2 Score: {score}")
        print(f"MAE: {mean_absolute_error(y_test, model.predict(X_test))}")

    def define_tensorflow_model(self):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        model = keras.Sequential([
            layers.InputLayer(shape=(33,)),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(6, activation='linear')
        ])

        def weighted_mse(y_true, y_pred):
            # Per-output squared error
            error = tf.square(y_true - y_pred)
            
            # Apply output weights
            weighted_error = error * output_weights
            
            # Mean over outputs, then mean over batch
            return tf.reduce_mean(weighted_error)

        output_weights = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=tf.float32)
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model
    
    def train_tensorflow_model(self, model, features, labels):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.callbacks import ReduceLROnPlateau

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=20,
            min_delta=1e-4,
            restore_best_weights=True
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6
)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, sample_weight=None, callbacks=[early_stopping, lr_scheduler])
        test_loss, test_mae = model.evaluate(X_test, y_test)
        print(f"Model Shape: {model.summary()}")
        print(f"Test MAE: {test_mae}")
        print(f"Test Loss: {test_loss}")
        print(f"R^2 Score: {r2_score(y_test, model.predict(X_test))}")

        
        

if __name__ == "__main__":
    engine = ModelsEngine()
    tf_model = engine.define_tensorflow_model()
    features, labels = engine.data_processor.load_features_and_labels()
    processed_features = engine.preprocessor.preprocess_features(features)
    processed_labels = engine.preprocessor.preprocess_labels(labels)
    engine.train_tensorflow_model(tf_model, processed_features, processed_labels)
    engine.run_modeling_pipeline()