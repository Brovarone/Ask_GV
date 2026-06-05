from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from .models import Target
from .utils import read_text

def load_config(config_path: Path) -> Dict[str, Any]:
    return json.loads(read_text(config_path))

def validate_targets_and_summary(cfg: dict) -> bool:
    """
    Verifica che tutti i modelli usati nei target e nei summary
    siano presenti nella sezione "models" della configurazione.

    Restituisce True se tutti i controlli hanno successo,
    altrimenti lancia ValueError e restituisce False.
    """
    # 1️⃣ Costruisci una mappa modello → provider (più veloce)
    models_dict = {
        m["model"]: m["provider"] for m in cfg.get("models", [])
    }

    # 2️⃣ Controllo dei target
    for t in cfg.get("targets", []):
        model_name = t.get("model")
        provider   = t.get("provider")
        if not model_name or not provider:
            raise ValueError(
                f"Target mancante 'model' o 'provider' in entry: {t}"
            )
        if model_name not in models_dict:
            raise ValueError(
                f"Modello target '{model_name}' non trovato nella sezione 'models'."
            )
        # opzionale: verifica che il provider indicato corrisponda a quello
        # definito per quel modello
        if models_dict[model_name] != provider:
            raise ValueError(
                f"Provider mismatch: target usa '{provider}' ma il modello "
                f"'{model_name}' è definito come '{models_dict[model_name]}'."
            )

    # -----------------------------------------------------------------
    # 3️⃣  Controllo dei SUMMARY
    # -----------------------------------------------------------------
    # La configurazione può contenere:
    #   * una lista di dict   ->  [ {...}, {...} ]
    #   * un singolo dict    ->  { "provider": "...", "model": "..." }
    # Normalizziamo sempre in una lista di dict.
    summary_cfg = cfg.get("summary")
    if isinstance(summary_cfg, dict):
        # un solo record
        summary_items = [summary_cfg]
    elif isinstance(summary_cfg, (list, tuple)):
        summary_items = list(summary_cfg)          # già una lista
    else:
        # valore inaspettato → trattiamo come lista vuota
        summary_items = []

    for s in summary_items:
        model_name = s.get("model")
        provider   = s.get("provider")

        if not model_name or not provider:
            raise ValueError(f"Summary entry missing 'model' or 'provider': {s}")

        if model_name not in models_dict:
            raise ValueError(f"Summary model '{model_name}' not present in 'models'.")

        if models_dict[model_name] != provider:
            raise ValueError(
                f"Provider mismatch for summary model '{model_name}': "
                f"expected '{models_dict[model_name]}', got '{provider}'."
            )

    # Tutti i controlli sono passati
    return True

def targets_from_config(cfg: Dict[str, Any]) -> List[Target]:
    out: List[Target] = []
    for t in cfg.get("targets", []):
        if not t.get("enabled", True):
            continue        

        out.append(Target(
            provider=t["provider"],
            model=t["model"],
            profile=t.get("profile", "systems_designer"),
            temperature=t.get("temperature", 0.4),
            max_tokens=t.get("max_tokens"),
            enabled=t.get("enabled", True),
            label=t.get("label"),
        ))
    return out
