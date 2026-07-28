param(
    [string]$Root = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
)

$ErrorActionPreference = "Stop"
$forbiddenExtensions = @(
    ".dcm", ".dicom", ".nii", ".gz", ".nrrd", ".mha", ".mhd",
    ".pth", ".pt", ".ckpt", ".onnx", ".safetensors", ".h5", ".hdf5"
)
$forbiddenNames = @(".env", "nohup.out")
$files = Get-ChildItem -LiteralPath $Root -Recurse -File -Force |
    Where-Object { $_.FullName -notmatch "\\.git\\" }

$blocked = $files | Where-Object {
    $_.Name -in $forbiddenNames -or
    $_.Extension.ToLowerInvariant() -in $forbiddenExtensions -or
    $_.Length -gt 10MB
}

$textFiles = $files | Where-Object {
    $_.Name -ne "public_release_check.ps1" -and
    $_.Extension -in ".py", ".sh", ".ps1", ".yaml", ".yml", ".json", ".md", ".txt", ".toml", ".example"
}
$patterns = @(
    "sk-[A-Za-z0-9]{16,}",
    "(?i)(api[_-]?key|access[_-]?token|secret[_-]?key)\s*=\s*['""][^'""]+",
    "(?i)[A-Z]:\\(?:Users|code|data|base model|15shot|chexfound|Anaconda|Windows)",
    "/home/[^<\s]+",
    "/mnt/[a-z]/"
)
$matches = $textFiles | Select-String -Pattern $patterns -ErrorAction SilentlyContinue

if ($blocked) {
    Write-Error "Blocked binary, medical-data, weight, or large files were found:`n$($blocked.FullName -join "`n")"
}
if ($matches) {
    $locations = $matches | ForEach-Object { "$($_.Path):$($_.LineNumber)" }
    Write-Error "Potential secrets or machine-specific paths were found:`n$($locations -join "`n")"
}

Write-Output "Public release check passed: $($files.Count) files."
