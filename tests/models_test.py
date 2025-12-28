from src.models import Preprocessor, ModelsEngine

def test_run_modeling_pipeline():
    engine = ModelsEngine()
    engine.run_modeling_pipeline()

def test_define_tensorflow_model():
    engine = ModelsEngine()
    model = engine.define_tensorflow_model()
    assert model is not None
    assert model.count_params() > 0
