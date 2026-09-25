from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple, Type


BASELINE_REGISTRY: Dict[str, Tuple[str, str]] = {
    "CNFGNN": ("privacy.baseline_adapters.cnfgnn", "CNFGNNPrivacyAdapter"),
    "FCGCN": ("privacy.baseline_adapters.fcgcn", "FCGCNPrivacyAdapter"),
    "FCFedGCN": ("privacy.baseline_adapters.fcfedgcn", "FCFedGCNPrivacyAdapter"),
    "FGNNEH": ("privacy.baseline_adapters.fgnneh", "FGNNEHPrivacyAdapter"),
    "Fed4TP": ("privacy.baseline_adapters.fed4tp", "Fed4TPPrivacyAdapter"),
    "FedAGAT": ("privacy.baseline_adapters.fedagat", "FedAGATPrivacyAdapter"),
    "ASTGAT": ("privacy.baseline_adapters.fedagat", "FedAGATPrivacyAdapter"),
    "FedGODE": ("privacy.baseline_adapters.fedgode", "FedGODEPrivacyAdapter"),
    "FedGRU": ("privacy.baseline_adapters.fedgru", "FedGRUPrivacyAdapter"),
    "SFL_RNN": ("privacy.baseline_adapters.sfl_rnn", "SFLRNNPrivacyAdapter"),
    "FedGTP": ("privacy.baseline_adapters.fedgtp", "FedGTPPrivacyAdapter"),
    "FedMetro": ("privacy.baseline_adapters.fedmetro", "FedMetroPrivacyAdapter"),
    "FedmSSA": ("privacy.baseline_adapters.fedmssa", "FedmSSAPrivacyAdapter"),
    "FedOSTC": ("privacy.baseline_adapters.fedostc", "FedOSTCPrivacyAdapter"),
    "FedSTG": ("privacy.baseline_adapters.fedstg", "FedSTGPrivacyAdapter"),
    "FedSTN": ("privacy.baseline_adapters.fedstn", "FedSTNPrivacyAdapter"),
    "FedTPS": ("privacy.baseline_adapters.fedtps", "FedTPSPrivacyAdapter"),
    "FedTSE": ("privacy.baseline_adapters.fedtse", "FedTSEPrivacyAdapter"),
    "FUELS": ("privacy.baseline_adapters.fuels", "FUELSPrivacyAdapter"),
    "ISTGNN": ("privacy.baseline_adapters.istgnn", "ISTGNNPrivacyAdapter"),
    "pFedCTP": ("privacy.baseline_adapters.pfedctp", "PFedCTPPrivacyAdapter"),
    "REFOL": ("privacy.baseline_adapters.refol", "REFOLPrivacyAdapter"),
    "SFL": ("privacy.baseline_adapters.sfl", "SFLPrivacyAdapter"),
    "STAGCN-EC": ("privacy.baseline_adapters.stagcn_ec", "STAGCNECPrivacyAdapter"),
    "STAGCN_EC": ("privacy.baseline_adapters.stagcn_ec", "STAGCNECPrivacyAdapter"),
    "STFAM": ("privacy.baseline_adapters.stfam", "STFAMPrivacyAdapter"),
    "STGCN": ("privacy.baseline_adapters.stagcn_ec", "STAGCNECPrivacyAdapter"),
    "TDLR": ("privacy.baseline_adapters.tdlr_sedlr", "TDLRSEDLRPrivacyAdapter"),
    "SEDLR": ("privacy.baseline_adapters.tdlr_sedlr", "TDLRSEDLRPrivacyAdapter"),
    "TDLR_SEDLR": ("privacy.baseline_adapters.tdlr_sedlr", "TDLRSEDLRPrivacyAdapter"),
    "2MGTCN": ("privacy.baseline_adapters.twomgtcn", "TwoMGTCNPrivacyAdapter"),
    "TwoMGTCN": ("privacy.baseline_adapters.twomgtcn", "TwoMGTCNPrivacyAdapter"),
    "UFCL": ("privacy.baseline_adapters.ufcl", "UFCLPrivacyAdapter"),
}


def _resolve_adapter(model_name: str) -> Type:
    module_name, class_name = BASELINE_REGISTRY[model_name]
    module = import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Adapter module {module_name!r} was imported, but class "
            f"{class_name!r} was not found. Please check that the adapter file "
            f"defines the expected class."
        ) from exc


def get_adapter(model_name: str):
    if model_name not in BASELINE_REGISTRY:
        raise KeyError(
            f"Privacy adapter for {model_name!r} is not implemented yet. "
            f"Available adapters: {sorted(BASELINE_REGISTRY)}"
        )
    return _resolve_adapter(model_name)()
