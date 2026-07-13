$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python -m pip install --upgrade "PyInstaller>=6.14,<7"
python -m PyInstaller --noconfirm --clean packaging/LEAPS-windows.spec

$DeploymentDir = Get-Item -Path "dist/LEAPS" -ErrorAction SilentlyContinue
if (!$DeploymentDir) {
    throw "A standalone LEAPS deployment directory was not produced under dist"
}
$Executable = Get-Item -Path (Join-Path $DeploymentDir.FullName "LEAPS.exe") -ErrorAction SilentlyContinue
if (!$Executable) {
    throw "The LEAPS application executable was not produced in $($DeploymentDir.FullName)"
}

& $Executable.FullName --packaging-self-test
if ($LASTEXITCODE -ne 0) {
    throw "The packaged LEAPS runtime self-test failed"
}

New-Item -ItemType Directory -Force -Path artifacts | Out-Null
$SourceDir = $Executable.DirectoryName
if ($env:WINDOWS_CERTIFICATE_PATH) {
    signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
        /f $env:WINDOWS_CERTIFICATE_PATH /p $env:WINDOWS_CERTIFICATE_PASSWORD `
        $Executable.FullName
    signtool verify /pa /v $Executable.FullName
}
$IsccCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($IsccCommand) {
    $IsccPath = $IsccCommand.Source
} else {
    $IsccPath = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6/ISCC.exe"
    if (!(Test-Path $IsccPath)) {
        throw "Inno Setup 6 was not found on the Windows runner"
    }
}
& $IsccPath "/DSourceDir=$SourceDir" packaging/leaps.iss
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
