# Codex Session Notes (Assistant-only)

Last updated: Fri Feb 13 2026

Purpose: scratchpad of facts gathered this session so future Codex launches retain context without re-reading large docs.

## Key Files Already Reviewed
- `PLAN_VJEPA2_E2E_AD.md`: describes Phase 1 joint JEPA + trajectory training; now notes ViT-L-as-default on A40s and ViT-g as a later scale-up.
- `CODEX_ENVIRONMENT_SETUP.md`: documents GPU/venv workflow plus dataset + ViT-L checkpoint verification.
    - `configs/train/vitg16/driving-joint-256px-16f.yaml`: canonical Phase 1 config (already set to ViT-L + `train_shifted_for_validation.csv`); pass it directly to `run_phase1_training.sh`.
- `scripts/run_phase1_training.sh`: wraps config rewrite + distributed launch.

## Environment / Infra Notes
- System has 8× NVIDIA A40 GPUs (driver 590.48.01, CUDA 13.1). `nvidia-smi` works only when Codex sandbox is `danger-full-access`.
- `codex-launch.sh` now defaults to `CODEX_SANDBOX_MODE=danger-full-access`; rerun `./codex-launch.sh` to start a GPU-capable session.
- `.venv` exists at repo root and should be activated automatically by the launcher.
- A40 memory budget cannot comfortably host ViT-g, so Phase 1 runs ViT-L (`checkpoints/vitl.pt`).
- NVIDIA AV CSV regenerated on Feb 13 via `scripts/prepare_nvidia_av_data.py` (now pulls 240 clips). With per-GPU batch size 3 and 8 ranks, `iterations per epoch` is now ≥10; we no longer need the `--batch-size 1` workaround unless we slim the dataset again.
- Always check for stray `python -m app.main --fname ...` processes before relaunching; lingering runs caused port collisions (`EADDRINUSE` on TCP 37129) and cascading NCCL socket aborts.
- Set `TORCH_DISTRIBUTED_BACKEND=gloo` before launching if NCCL keeps timing out (e.g., `WorkNCCL(... OpType=ALLGATHER ...) timeout`). `src/utils/distributed.py` now respects the env var; default remains NCCL.

## Latest Phase 1 Run Attempts
- `logs/vjepa_drive/phase1-20260213-001210` (00:12 UTC): first ViT-L launch, `batch_size=3`. Every rank logged `iterations per epoch/dataset length: 0/0` and loop never advanced. Terminated after confirming data issue.
- `logs/vjepa_drive/phase1-20260213-001453` (00:14 UTC): rerun while previous processes still alive. Rank0 fell back to single-process and trained alone (loss dropped 35→19 over epochs 41–43) but ranks 1–7 stayed at 0/0 and crashed during DDP init with `torch.distributed.DistBackendError: NCCL ... socketStartConnect ... Software caused connection abort`. Killed remaining PIDs at 00:16:13.
- `logs/vjepa_drive/phase1-20260213-002830` (00:28 UTC): first ViT-L run on the 240-clip CSV + default batch size 3. All ranks reported `iterations per epoch/dataset length: 10/10`, but stdout wasn’t saved and the job eventually died without a trace.
- `logs/vjepa_drive/phase1-20260213-005035` (00:50 UTC): rerun with `console.log`. After ~10 minutes the NCCL watchdog killed the job during WorkNCCL SeqNum=1 ALLGATHER (600 s timeout). Killed at ~01:01 UTC.
- `logs/vjepa_drive/phase1-20260213-010350` (01:03 UTC): current run launched with `TORCH_DISTRIBUTED_BACKEND=gloo`, so all output streams into `console.log` and we avoid the NCCL watchdog. Monitor this folder for loss curves/checkpoints.

## Next Session Checklist
1. Always mount GPUs via `./codex-launch.sh` (unrestricted sandbox) before touching training.
2. Verify inputs exist: `checkpoints/vitl.pt` and `data/nvidia_av/train_shifted_for_validation.csv`.
3. Use `configs/train/vitg16/driving-joint-256px-16f.yaml` (patched for ViT-L + single CSV) when calling  
   `scripts/run_phase1_training.sh --gpus 0,1,2,3,4,5,6,7 --config configs/train/vitg16/driving-joint-256px-16f.yaml --pretrained checkpoints/vitl.pt --datasets data/nvidia_av/train_shifted_for_validation.csv` (per-GPU batch size stays 3 now that the CSV has 240 rows).
4. Capture logs + tensorboard dir after launch; note any failures plus GPU utilization.
5. If training already running, prioritize monitoring/triage over new launches; update this file with outcomes.

Keep this file concise; update timestamps + bullet facts as new context emerges.
