# Codex Environment Setup

Use these steps at the start of each new Codex session in this repository.

## 0) If `.venv` does not exist yet (one-time bootstrap)

```bash
cd /localhome/local-samehm/vjepa2
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e physical_ai_av
```

## 1) Go to repo root

```bash
cd /localhome/local-samehm/vjepa2
```

## 2) Activate the project venv

```bash
source .venv/bin/activate
```

## 3) Verify you are using the venv

```bash
which python
python --version
python -c "import sys; print(sys.prefix)"
```

Expected: `which python` should point to `/localhome/local-samehm/vjepa2/.venv/bin/python`.

## 4) Restarting this Codex session later

Any time you need to resume work with the same GPU/venv setup:

```bash
cd /localhome/local-samehm/vjepa2
./codex-launch.sh
```

The launcher now defaults to `--sandbox unrestricted`, so GPUs are visible inside Codex without extra steps. Export `CODEX_MODEL`, `CODEX_SANDBOX_MODE`, or other `CODEX_*` vars before running the script if you need overrides.

## 4) Inspect available GPUs (expect 8x A40)

```bash
nvidia-smi
```

You should see eight NVIDIA A40 GPUs listed. If `nvidia-smi` reports `Failed to initialize NVML`, capture the exact error in your run notes and notify the infra owner before launching training.

## 5) Run project scripts with the active env

Example:

```bash
python scripts/prepare_nvidia_av_data.py
```

## 6) Verify dataset + checkpoint inputs

Phase 1 now trains ViT-L on a single NVIDIA AV CSV selected for A40 memory limits:

```bash
ls /localhome/local-samehm/vjepa2/checkpoints/vitl.pt
ls /localhome/local-samehm/vjepa2/data/nvidia_av/train_shifted_for_validation.csv
```

If either file is missing, sync it from shared storage before launching jobs.

## 7) Phase 1 config + launch template (A40 default)

`configs/train/vitg16/driving-joint-256px-16f.yaml` already points to ViT-L + the single CSV. Launch jobs with:

```bash
scripts/run_phase1_training.sh \
  --fname configs/train/vitg16/driving-joint-256px-16f.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --pretrained checkpoints/vitl.pt \
  --datasets data/nvidia_av/train_shifted_for_validation.csv
```

Append any extra flags (resume, logging overrides, etc.) as needed, but keep ViT-L and the dataset path unchanged unless coordinated.

> NOTE: `scripts/prepare_nvidia_av_data.py` now downloads 240 clips, so the default batch size of 3 per GPU works across 8 ranks. If you trim the CSV later, ensure `world_size × batch_size` does not exceed the number of rows or update the launch args accordingly.

## 8) Optional: run without activating shell

```bash
/localhome/local-samehm/vjepa2/.venv/bin/python scripts/prepare_nvidia_av_data.py
```

## 9) Deactivate when done

```bash
deactivate
```
