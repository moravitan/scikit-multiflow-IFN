import os
import shutil
from skml.IOLIN import PureMultiple
from skml import IfnClassifier


alpha = 0.99
test_tmp_folder = "tmpPureMultiple"


def _setup_test_env():
    if not os.path.isdir(test_tmp_folder):
        os.mkdir(test_tmp_folder)


def _clean_test_env():
    shutil.rmtree(test_tmp_folder, ignore_errors=True)


def test_pure_multiple():
    _setup_test_env()
    ifn = IfnClassifier(alpha)
    pure_IOLIN = PureMultiple(ifn, test_tmp_folder, n_min=0, n_max=60, Pe=0.7)
    chosen_model = pure_IOLIN.pure_model_generation()
    assert chosen_model is not None
    _clean_test_env()


test_pure_multiple()