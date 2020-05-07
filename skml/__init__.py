
from .ifn_classifier import IfnClassifier
from .IOLIN.meta_learning import MetaLearning
from .IOLIN.olin import OnlineNetwork
from .IOLIN.ifn_regenerative import OnlineNetworkRegenerative
from .IOLIN.ifn_basic_incremental import BasicIncremental
from .IOLIN.ifn_pure_multiple_model import PureMultiple

from ._version import __version__

__all__ = ['IfnClassifier', 'OnlineNetwork', 'MetaLearning', 'OnlineNetworkRegenerative', 'BasicIncremental',
           'PureMultiple',
           '__version__']

