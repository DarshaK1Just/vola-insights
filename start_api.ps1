# Start the TransactionRAG FastAPI server
# --reload-dir ensures WatchFiles ONLY watches src/ and api/
# and never triggers on .venv/ package changes

$env:PYTHONUTF8 = '1'

Write-Host "Starting FinanceAI API server..." -ForegroundColor Cyan
Write-Host "API:     http://localhost:8000"     -ForegroundColor Green
Write-Host "Docs:    http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""

.\.venv\Scripts\python -m uvicorn api.app:app `
    --host 0.0.0.0 `
    --port 8000 `
    --reload `
    --reload-dir src `
    --reload-dir api `
    --log-level warning
