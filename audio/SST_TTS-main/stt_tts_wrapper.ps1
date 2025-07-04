param(
    [Parameter(Mandatory=$true)]
    [string]$Text,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFile = "D:\piper_tts\output\speech.wav"
)

# CORRECTED PATHS based on your actual setup
$piperPath = "D:\piper_tts\piper_models\piper\piper.exe"
$modelPath = "D:\piper_tts\piper_models\en_US-lessac-medium.onnx"
$tempDir = "D:\piper_tts\temp"
$outputDir = Split-Path $OutputFile -Parent

# Create directories if they don't exist
if (-not (Test-Path $tempDir)) { New-Item -ItemType Directory -Force -Path $tempDir | Out-Null }
if (-not (Test-Path $outputDir)) { New-Item -ItemType Directory -Force -Path $outputDir | Out-Null }

# Validate paths
if (-not (Test-Path $piperPath)) {
    Write-Error "Piper executable not found at: $piperPath"
    exit 1
}

if (-not (Test-Path $modelPath)) {
    Write-Error "Model file not found at: $modelPath"
    exit 1
}

# Also check for the model config file
$modelConfigPath = $modelPath + ".json"
if (-not (Test-Path $modelConfigPath)) {
    Write-Warning "Model config file not found at: $modelConfigPath (this might cause issues)"
}

# Create temporary text file
$tempFile = Join-Path $tempDir "temp_$(Get-Random).txt"
$Text | Out-File -FilePath $tempFile -Encoding UTF8 -NoNewline

try {
    Write-Host "Running Piper TTS..."
    Write-Host "Piper: $piperPath"
    Write-Host "Model: $modelPath"
    Write-Host "Output: $OutputFile"
    Write-Host "Input text: $Text"
    
    # Use simpler approach with cmd
    $arguments = "--model `"$modelPath`" --output_file `"$OutputFile`""
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $piperPath
    $processInfo.Arguments = $arguments
    $processInfo.RedirectStandardInput = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $process.Start()
    
    # Send text to stdin
    $process.StandardInput.WriteLine($Text)
    $process.StandardInput.Close()
    
    # Wait for completion
    $process.WaitForExit()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    
    Write-Host "Process exit code: $($process.ExitCode)"
    if ($stdout) { Write-Host "Stdout: $stdout" }
    if ($stderr) { Write-Host "Stderr: $stderr" }
    
    if ($process.ExitCode -eq 0 -and (Test-Path $OutputFile)) {
        Write-Host "TTS Success: Audio generated at $OutputFile"
        
        # Play the audio
        try {
            Start-Process -FilePath $OutputFile -WindowStyle Hidden
        } catch {
            Write-Host "Could not auto-play audio, but file was created successfully"
        }
        
        exit 0
    } else {
        Write-Error "TTS generation failed with exit code: $($process.ExitCode)"
        exit 1
    }
    
} catch {
    Write-Error "TTS Error: $($_.Exception.Message)"
    exit 1
} finally {
    # Clean up temp file
    if (Test-Path $tempFile) {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }
}