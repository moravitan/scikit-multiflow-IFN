from skml.IOLIN import MultipleModel
from skml import IfnClassifier
from skmultiflow.data import SEAGenerator
import numpy as np

alpha = 0.99


def test_multiple_model(tmpdir):
    dir = tmpdir.mkdir("tmpMultipleModel")
    ifn = IfnClassifier(['float64', 'float64', 'float64'], alpha)
    stream_generator = SEAGenerator(random_state=23)
    multiple_model = MultipleModel(ifn, dir, n_min=0, n_max=200, Pe=0.7, data_stream_generator=stream_generator)
    last_model = multiple_model.generate()
    expected_number_of_generated_models = 3
    number_of_generated_models = multiple_model.counter - 1

    stream_generator = SEAGenerator(random_state=23)
    X, y = stream_generator.next_sample(multiple_model.window)
    predictions = multiple_model.classifier.predict(X)

    expected_predictions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                            1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                            0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                            1.0, 0.0, 1.0, 0.0]

    correct_predictions = [i for i, j in zip(y, predictions) if i == j]
    expected_correct_predictions = 38

    performance = len(correct_predictions) / len(y)
    expected_performance = 0.95

    assert last_model is not None
    assert number_of_generated_models == expected_number_of_generated_models
    assert np.alltrue(predictions == expected_predictions)
    assert expected_performance == performance
    assert len(correct_predictions) == expected_correct_predictions
