REGISTRY = {}

from .basic_controller import BasicMAC
from .cacom_controller import CACOM_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cacom_mac"] = CACOM_MAC
