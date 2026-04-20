# start.ps1 в bin/
$originalDir = Get-Location

$bin = Split-Path -Parent $MyInvocation.MyCommand.Definition

# корневая папка проекта (один уровень выше bin)
$root = Split-Path $bin -Parent
Set-Location $root

function Get-Config {
    param(
        [string]$ConfigPath = "$bin\config.json"
    )

    if (-Not (Test-Path $ConfigPath)) {
        throw "Config file not found: $ConfigPath"
    }

    $json = Get-Content $ConfigPath -Raw
    return $json | ConvertFrom-Json
}

$config = Get-Config
$llmPath = $config.models.llm
$kobold = $config.kobold
$uvicorn = $config.uvicorn
$streamlit = $config.streamlit

$koboldExe = $kobold.exe

$koboldArgs = @(
    "--port", $kobold.port,
    "--host", $kobold.host,
    "`"$llmPath`""
)

Write-Host "Starting KoboldCpp..."
Start-Process -FilePath $koboldExe -ArgumentList $koboldArgs

Write-Host "Sleeping 20 seconds to let KoboldCpp start..."
Start-Sleep -Seconds 20

Write-Host "Starting FastAPI..."
$commandFastApi = "uvicorn scripts.server.server:app --host $($uvicorn.host) --port $($uvicorn.port)"
Write-Host $commandFastApi
Start-Process powershell -ArgumentList "-NoExit", "-Command", $commandFastApi

Write-Host "Starting Streamlit UI..."
$commandStreamlit = "python -m streamlit run $root/scripts/ui/ui.py --server.address=$($streamlit.host) --server.port=$($streamlit.port) --server.headless=true --browser.gatherUsageStats=false"
Write-Host $commandStreamlit
Start-Process powershell -ArgumentList "-NoExit", "-Command", $commandStreamlit

Write-Host "All processes started."
Write-Host "KoboldCpp port: $($kobold.port)"
Write-Host "FastAPI URL:  http://$($uvicorn.host):$($uvicorn.port)"
Write-Host "Streamlit URL: http://$($streamlit.host):$($streamlit.port)"

Set-Location $originalDir