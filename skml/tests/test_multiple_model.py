import pytest
from skml.IOLIN import MultipleModel
from skml import IfnClassifier

alpha = 0.99
test_tmp_folder = "tmpMultipleModel"


def test_pure_multiple(tmpdir):
    dir = tmpdir.mkdir("tmpMultipleModel")
    ifn = IfnClassifier(alpha)
    multiple_model = MultipleModel(ifn, dir, n_min=0, n_max=200, Pe=0.7)
    chosen_model = multiple_model.generate()
    assert chosen_model is not None
    assert len(dir.listdir()) == 4
