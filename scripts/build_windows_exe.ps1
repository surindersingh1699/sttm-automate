$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..").Path
$EntryScript = Join-Path $RootDir "scripts\windows_app_entry.py"
$AppName = "STTM-Automate"

Write-Host "[build-win] Python version:"
python --version

Write-Host "[build-win] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r (Join-Path $RootDir "requirements.txt")
python -m pip install pyinstaller

Write-Host "[build-win] Building .exe..."
pyinstaller `
  --noconfirm `
  --onefile `
  --windowed `
  --name $AppName `
  --paths $RootDir `
  --add-data "$RootDir\static;static" `
  --collect-all faster_whisper `
  --collect-all webview `
  $EntryScript

Write-Host "[build-win] Build complete."
Write-Host ("[build-win] Output: {0}" -f (Join-Path $RootDir "dist\$AppName.exe"))
