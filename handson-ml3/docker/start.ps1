# Start Docker Desktop if needed, start the container and open JupyterLab (with token).
Set-Location $PSScriptRoot
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
    do { Start-Sleep 3; docker info *> $null } until ($LASTEXITCODE -eq 0)
}
docker compose up -d
$url = $null
for ($i = 0; $i -lt 30 -and -not $url; $i++) {
    Start-Sleep 2
    $url = cmd /c "docker logs handson-ml3 2>&1" | Select-String -Pattern 'http://127\.0\.0\.1:8888/lab\?token=\w+' | Select-Object -Last 1 | ForEach-Object { $_.Matches[0].Value }
}
if ($url) { Start-Process $url } else { "token not found; check: docker logs handson-ml3" }
