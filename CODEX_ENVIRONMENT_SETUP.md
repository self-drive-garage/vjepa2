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

## 4) Run project scripts with the active env

Example:

```bash
python scripts/prepare_nvidia_av_data.py
```

## 5) Optional: run without activating shell

```bash
/localhome/local-samehm/vjepa2/.venv/bin/python scripts/prepare_nvidia_av_data.py
```

## 6) Deactivate when done

```bash
deactivate
```
