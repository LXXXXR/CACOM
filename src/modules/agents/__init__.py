REGISTRY = {}

from .rnn_agent import RNNAgent
from .cacom_agent import CACOM_Agent

REGISTRY["rnn"] = RNNAgent
REGISTRY["cacom"] = CACOM_Agent
