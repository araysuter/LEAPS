$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python -m pip install --upgrade build "Nuitka>=2.7"
pyside6-deploy -c pysidedeploy.spec --force

$Executable = Get-ChildItem -Path dist -Recurse -Filter LEAPS.exe | Select-Object -First 1
if (!$Executable) {
    throw "A LEAPS.exe deployment was not produced under dist"
}

New-Item -ItemType Directory -Force -Path artifacts | Out-Null
$SourceDir = $Executable.DirectoryName
if ($env:WINDOWS_CERTIFICATE_PATH) {
    signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
        /f $env:WINDOWS_CERTIFICATE_PATH /p $env:WINDOWS_CERTIFICATE_PASSWORD `
        $Executable.FullName
    signtool verify /pa /v $Executable.FullName
}
ISCC.exe "/DSourceDir=$SourceDir" packaging/leaps.iss
$Installer = "artifacts/LEAPS-Windows-x64-Setup.exe"
if (!(Test-Path $Installer)) {
    throw "The Windows installer was not produced"
}

if ($env:WINDOWS_CERTIFICATE_PATH) {
    signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
        /f $env:WINDOWS_CERTIFICATE_PATH /p $env:WINDOWS_CERTIFICATE_PASSWORD `
        $Installer
    signtool verify /pa /v $Installer
}
