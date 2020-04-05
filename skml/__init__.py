
from ._ifn_Classifier import IfnClassifier
from .IOLIN._Meta_Learning import MetaLearning
from .IOLIN._Regenerative import OnlineNetworkRegenerative
from .IOLIN._Basic_Incremental import BasicIncremental

from ._version import __version__

__all__ = ['_ifn_Classifier', '_Meta_Learning', 'OnlineNetworkRegenerative', 'BasicIncremental',
           '__version__']
