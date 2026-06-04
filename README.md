# RPG LLM Modular Clean

Pipeline Python modulare per:
- leggere una repo GitHub o una lista di file `.md`
- costruire un knowledge pack delle regole
- interrogare più LLM con profili diversi
- confrontare e salvare i risultati

## Requisiti

- Python 3.11+
- `requests`
- chiavi API opzionali in base ai provider usati

## Variabili ambiente

- `OPENROUTER_API_KEY`
- `GEMINI_API_KEY` oppure `GOOGLE_API_KEY`
- `PERPLEXITY_API_KEY`
- `NVIDIA_API_KEY` oppure `NIM_API_KEY`

## Esempio rapido

```bash
python -m rpg_llm_modular_clean.cli \
  --config rpg_llm_modular_clean/config.example.json \
  --files rules/*.md \
  --build-only
```

Per una run completa:

```bash
python -m rpg_llm_modular_clean.cli \
  --config rpg_llm_modular_clean/config.example.json \
  --files rules/*.md \
  --question "Come miglioro il bilanciamento della magia ai primi livelli?"
```

## Struttura

- `cli.py`: entrypoint CLI
- `ingest.py`: input repo/file markdown
- `retrieval.py`: chunking e ranking contesto
- `summary.py`: summary tecnico iniziale
- `pipeline.py`: orchestrazione multi-provider
- `reporting.py`: report e CSV/JSON
- `providers/`: adapter API provider
- `observability/`: logging strutturato
