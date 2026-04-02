$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
$env:PYTHONPATH = (Join-Path (Get-Location) "python-sdk")
python ".\python-sdk\examples\stand.py"
