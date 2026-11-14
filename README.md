# Atheer: Spatial-CSI Astronaut Health & Safety Guard

Contactless, camera-free health and safety monitoring for space habitats, built from the Wi-Fi signals a station already has.
NASA Space Apps proof of concept: published datasets, published models, a thin AI fusion layer, and a mission-control dashboard.

```
public CSI datasets ─► signal processing + pretrained models ─► fusion & alerts ─► replay ─► dashboard
 WiRe-Breath, UT-HAR    breathing rate, activity, fall, motion    rules + LLM brief   FastAPI   React
```

## Run the demo

```bash
pip install -e .            # Python 3.10+
cd dashboard && npm install && npm run build && cd ..
uvicorn atheer.api.main:app --port 8010
# open http://127.0.0.1:8010  →  Start replay
```

Development with hot reload: run the uvicorn command above, then `cd dashboard && npm run dev` and open http://localhost:3000 (Next.js proxies `/api` to the backend on :8010; the telemetry WebSocket connects to :8010 directly).
Set `ANTHROPIC_API_KEY` to have Claude write the crew brief; without it a rule-based summary is used.

## Rebuild data and results

```bash
python scripts/build_tracks.py --synthetic                         # placeholders (no downloads)
python scripts/train_breath_confidence.py --root data/raw/wirebreath # breathing eval + confidence calibration
python scripts/build_tracks.py --wirebreath data/raw/wirebreath    # real breathing tracks (uses the model)
python scripts/build_tracks.py --uthar data/raw/sensefi            # real activity / fall tracks
python scripts/eval_respiration.py                                 # classic baseline -> results/eval_respiration_classic.json
python scripts/exp_activity_robust.py groups && python scripts/exp_activity_robust.py p1train --variant robust  # robust activity model
python scripts/eval_activity.py                                    # held-out recordings -> results/eval_activity.json, eval_anomaly.json
pytest -q
```

Data sources: WiRe-Breath GitHub release v1.0.0 (`data/raw/wirebreath`), SenseFi Google Drive (`UT_HAR.zip` into `data/raw/sensefi`, weights into `models/sensefi/UT-HAR`). See [THIRD_PARTY.md](THIRD_PARTY.md).

## Layout

| Path | What it does |
|---|---|
| `atheer/ingest/` | Dataset loaders (WiRe-Breath port of the authors' MATLAB loader) |
| `atheer/modules/respiration.py` | Amplitude + cross-subcarrier-ratio phase channels → band-pass → summed spectra of the best 30 channels → Viterbi tracking over spectral peaks → rate + calibrated confidence (classic PCA peak kept as baseline) |
| `atheer/modules/activity.py` | SenseFi ResNet18 trained with robustness augmentation on UT-HAR (published weights kept for reproduction); dead-subcarrier repair, per-window normalisation, calibrated probabilities |
| `atheer/modules/anomaly.py` | Motion index vs personal baseline, fall probability, optional IsolationForest |
| `atheer/fusion/engine.py` | Crew context, context-aware limits, confidence gating, debounced C&W alerts with evidence |
| `atheer/fusion/brief.py` | Crew brief from Claude, with an offline template fallback |
| `atheer/replay/` + `scenarios/` | Scenario timeline that streams dataset tracks through the pipeline |
| `atheer/api/main.py` | REST + WebSocket backend |
| `dashboard/` | Next.js 16 + React 19, Tailwind 4, shadcn/Radix, TanStack Query, Recharts and React Three Fiber; exported as static files to `dashboard/dist`. Pages: Crew, Station (3D), Shift log, How it works |

Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before presenting results. How Atheera maps to the Space Apps challenge and NASA's risks: [docs/CHALLENGE.md](docs/CHALLENGE.md).
