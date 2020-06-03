
from .ifn_classifier import IfnClassifier
from .IOLIN.meta_learning import MetaLearning
from .IOLIN.iolin import IncrementalOnlineNetwork
from .IOLIN.ifn_regenerative import OnlineNetworkRegenerative
from .IOLIN.ifn_basic_incremental import BasicIncremental
from .IOLIN.ifn_pure_multiple_model import PureMultiple


__all__ = ['IfnClassifier', 'IncrementalOnlineNetwork', 'MetaLearning', 'OnlineNetworkRegenerative', 'BasicIncremental',
           'PureMultiple']

