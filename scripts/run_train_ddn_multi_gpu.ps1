# ============================================================
# DDN Multi-GPU Launch Script (PowerShell / Windows)
# ============================================================
# Usage:
#   powershell -File scripts/run_train_ddn_multi_gpu.ps1 [-ConfigFile PATH] [-NumGpus N] [-OutputDir PATH]
#
# Example:
#   powershell -File scripts/run_train_ddn_multi_gpu.ps1
#   powershell -File scripts/run_train_ddn_multi_gpu.ps1 -NumGpus 4
#
# Notes:
#   - Uses torchrun for multi-GPU launch
#   - Model uses FSDP (SHARD_GRAD_OP)
#   - Total batch size = batch_size_per_gpu x NUM_GPUS
# ============================================================

param(
    [string]$ConfigFile = "chexfound/configs/train/lista16_ibot333_highres640.yaml",
    [int]$NumGpus = 0,
    [string]$OutputDir = "./outputs/ddn_multi_gpu"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$ProjectRoot;$env:PYTHONPATH"
} else {
    $ProjectRoot
}
Set-Location $ProjectRoot

# Auto-detect GPU count if not specified
if ($NumGpus -eq 0) {
    try {
        $NumGpus = (nvidia-smi -L 2>$null | Measure-Object).Count
        if ($NumGpus -eq 0) { $NumGpus = 1 }
    } catch {
        $NumGpus = 1
    }
}

if ($NumGpus -lt 2) {
    Write-Warning "NumGpus=$NumGpus, at least 2 GPUs recommended for multi-GPU training"
}

$RecWorkers = $NumGpus * 5
Write-Host "============================================"
Write-Host "  DDN Multi-GPU Training (Windows)"
Write-Host "  Config:      $ConfigFile"
Write-Host "  Num GPUs:    $NumGpus"
Write-Host "  Output Dir:  $OutputDir"
Write-Host "  Recommended num_workers: $RecWorkers (total)"
Write-Host "============================================"

$env:TOKENIZERS_PARALLELISM = "false"
$env:NVIDIA_TF32_OVERRIDE = "1"

torchrun `
    --standalone `
    --nproc_per_node=$NumGpus `
    chexfound/train/train.py `
    --config-file $ConfigFile `
    --output-dir $OutputDir `
    $args

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Training finished. Logs and checkpoints saved to: $OutputDir"
