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
DELL_LEGIBILITY_MODEL=/absolute/path/to/legibility_model_new.onnx
OLLAMA_MODEL=gemma4:e4b
OLLAMA_HOST=http://localhost:11434
```

## 4) Model Weights
Expected layout model file:
- `weights/layout_model_new.onnx`
- `weights/legibility_model_new.onnx`

You can either:
- keep these default paths, or
- point `DELL_LAYOUT_MODEL` / `DELL_LEGIBILITY_MODEL` to different locations.

## 5) Input and Run
Put images under:
- `input_images/`

If your images live in DigitalOcean Spaces, from the repo root run:
```bash
./scripts/sync_spaces_input_images.sh
```

Optional arguments:
```bash
# custom local destination directory
./scripts/sync_spaces_input_images.sh ./input_images

# sync only a prefix/folder from the bucket
./scripts/sync_spaces_input_images.sh ./input_images some/prefix
```

The script prompts for Spaces credentials securely and does not persist them to `~/.aws/credentials`.

Run:
```bash
python3 src/main_pipeline.py
```

Outputs are written to:
- `outputs/`

## 6) Docker (Build + Run)
This container installs all Python/system dependencies and runs the pipeline script.
Ollama is **not** installed in the container; it must be running on the host.

Build:
```bash
docker build -t newspaper-extractor:latest .
```

Run on macOS/Windows:
```bash
mkdir -p outputs .pipeline_state
docker run --rm \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$PWD/input_images:/app/input_images:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/.pipeline_state:/app/.pipeline_state" \
  newspaper-extractor:latest
```

Run on Linux:
```bash
mkdir -p outputs .pipeline_state
docker run --rm \
  --add-host host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$PWD/input_images:/app/input_images:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/.pipeline_state:/app/.pipeline_state" \
  newspaper-extractor:latest
```

## Appendix: Install AWS CLI on Ubuntu
These commands install AWS CLI v2 (official AWS installer):

```bash
sudo apt-get update
sudo apt-get install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

For ARM Ubuntu machines, use:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
```
