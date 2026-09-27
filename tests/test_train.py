from train import create_model


def test_create_model():
    model = create_model()

    assert model is not None
    assert "preprocessing" in model.named_steps
    assert "model" in model.named_steps