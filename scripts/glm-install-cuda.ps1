param([Parameter(Mandatory = $true)][string]$InstallDirectory)
$ErrorActionPreference = 'Stop'

# NVIDIA redistrib_12.8.1.json, Windows x86_64, retrieved 2026-10-20.
$Components = @(
    @('cuda_nvcc', '12.8.93', '9fdc70b4271ed9aad4d64cd7076a7d96ec36512d074b9995fe638de669197391'),
    @('cuda_cudart', '12.8.90', '4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b'),
    @('libcublas', '12.8.4.1', '57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa'),
    @('cuda_cccl', '12.8.90', 'bd8548fa1ae82f92910bebc3079e14bd58c5a92aa64596d46bd610a478cb39d7')
)
New-Item -ItemType Directory -Force $InstallDirectory | Out-Null
foreach ($Component in $Components) {
    $Name, $Version, $Expected = $Component
    $ArchiveName = "$Name-windows-x86_64-$Version-archive"
    $Archive = Join-Path $env:RUNNER_TEMP "$ArchiveName.zip"
    $Extract = Join-Path $env:RUNNER_TEMP $ArchiveName
    $Url = "https://developer.download.nvidia.com/compute/cuda/redist/$Name/windows-x86_64/$ArchiveName.zip"
    Invoke-WebRequest -Uri $Url -OutFile $Archive
    if ((Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant() -ne $Expected) {
        throw "NVIDIA archive verification failed for $Name"
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Extract
    Copy-Item -Path (Join-Path $Extract "$ArchiveName/*") -Destination $InstallDirectory -Recurse -Force
    Remove-Item -LiteralPath $Archive -Force
    Remove-Item -LiteralPath $Extract -Recurse -Force
}
"CUDA_PATH=$InstallDirectory" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
(Join-Path $InstallDirectory 'bin') | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
& (Join-Path $InstallDirectory 'bin/nvcc.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'nvcc version check failed' }
