
from .ifn_Classifier import IfnClassifier
from .IOLIN.Meta_Learning import MetaLearning
from .IOLIN.OLIN import OnlineNetwork
from .IOLIN.Regenerative import OnlineNetworkRegenerative
from .IOLIN.Basic_Incremental import BasicIncremental

from ._version import __version__

__all__ = ['IfnClassifier', 'OnlineNetwork', 'MetaLearning', 'OnlineNetworkRegenerative', 'BasicIncremental',
           '__version__']

