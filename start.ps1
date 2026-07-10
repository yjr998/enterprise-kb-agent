# PowerShell 启动脚本（等价于 start.bat）
Set-Location $PSScriptRoot

Write-Host "========================================"
Write-Host "  Enterprise KB Agent"
Write-Host "========================================"

if (-not (Test-Path "chroma_db")) {
    Write-Host "[提示] 正在构建向量库..."
    python scripts/build_vectordb.py
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$models = ollama list 2>$null
if ($models -notmatch "qwen2.5:3b") {
    Write-Host "未找到 qwen2.5:3b，正在拉取..."
    ollama pull qwen2.5:3b
}

Write-Host "[启动] http://127.0.0.1:8000"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
