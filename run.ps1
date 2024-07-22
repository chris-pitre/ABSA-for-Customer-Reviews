if (-Not (Test-Path .\.venv)){
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install .\en_core_web_lg-3.7.1.tar.gz .\en_core_web_sm-3.7.1.tar.gz
    python -m pip install -r .\requirements.txt
} else {
    .\.venv\Scripts\Activate.ps1 
}
if(Get-Command git -errorAction silentlyContinue){
    python -m streamlit run .\app.py
} else {
    Write-Output "ERROR Git not installed: Please install Git for Windows"
}
