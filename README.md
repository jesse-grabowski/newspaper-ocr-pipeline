# newspaper-article-extractor

Extract newspaper article text from local page images using:
- Dell ONNX layout detection
- Ollama vision-language OCR (default: `gemma4:e4b`)

## Quick Setup (New Machine)
1. Clone repo and enter it.
2. Create and activate a virtual environment.
3. Install Python dependencies.
4. Install/start Ollama and pull the OCR model.
5. Configure `.env`.
6. Add page images and run the pipeline.

## 1) Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Ollama
```bash
ollama pull gemma4:e4b
```

If Ollama is not already running, start it on the default host (`http://localhost:11434`).

## 3) Configure `.env`
```bash
cp .env.template .env
```

Then edit `.env`:
```bash
DELL_LAYOUT_MODEL=/absolute/path/to/layout_model_new.onnx
OLLAMA_MODEL=gemma4:e4b
OLLAMA_HOST=http://localhost:11434
```

## 4) Model Weights
Expected layout model file:
- `weights/layout_model_new.onnx`

You can either:
- keep this default path, or
- point `DELL_LAYOUT_MODEL` to a different location.

## 5) Input and Run
Put images under:
- `input_images/`

Run:
```bash
python3 src/main_pipeline.py
```

Outputs are written to:
- `outputs/`
