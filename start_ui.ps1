# Start the FinanceAI Streamlit frontend

$env:PYTHONUTF8 = '1'

Write-Host "Starting FinanceAI UI..." -ForegroundColor Cyan
Write-Host "UI: http://localhost:8501"  -ForegroundColor Green
Write-Host ""

.\.venv\Scripts\streamlit run frontend\app.py --server.port 8501
