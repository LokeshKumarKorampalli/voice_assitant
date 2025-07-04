# Working Batch TTS Script
Write-Host "=== Batch Text-to-Speech Processing ===" -ForegroundColor Cyan

# Create output directory
$outputDir = "D:\piper_tts\output"
if (-not (Test-Path $outputDir)) {
  New-Item -ItemType Directory -Path $outputDir -Force
  Write-Host "Created output directory: $outputDir" -ForegroundColor Green
}

# Define your texts here - EDIT THIS ARRAY
$texts = Get-Content "texts.txt"

Write-Host "Processing $($texts.Count) text samples..." -ForegroundColor Yellow

# Process each text
$successCount = 0
for ($i = 0; $i -lt $texts.Count; $i++) {
  $text = $texts[$i].Trim()
  if ($text -eq "") { 
    continue 
  }
    
  $filename = "audio_{0:D3}.wav" -f ($i + 1)
  $currentNum = $i + 1
  $totalNum = $texts.Count
    
  Write-Host "[$currentNum/$totalNum] Processing: $filename" -ForegroundColor Yellow
  Write-Host "  Text: $($text.Substring(0, [Math]::Min(60, $text.Length)))..." -ForegroundColor Gray
    
  # Change to piper directory
  $originalLocation = Get-Location
  Set-Location "D:\piper_tts\piper_models\piper"
    
  try {
    # Run piper conversion
    $outputPath = "..\..\output\$filename"
    $text | .\piper.exe --model ..\en_US-lessac-medium.onnx --output_file $outputPath
        
    # Check if file was created
    Set-Location $originalLocation
    $fullOutputPath = "$outputDir\$filename"
        
    if (Test-Path $fullOutputPath) {
      $fileSize = (Get-Item $fullOutputPath).Length
      Write-Host "  ✓ Success: $filename ($fileSize bytes)" -ForegroundColor Green
      $successCount++
    }
    else {
      Write-Host "  ✗ Failed: $filename (file not created)" -ForegroundColor Red
    }
  }
  catch {
    Set-Location $originalLocation
    Write-Host "  ✗ Error: $($_.Exception.Message)" -ForegroundColor Red
  }
}

Write-Host "`n=== Processing Complete ===" -ForegroundColor Cyan
Write-Host "Success Rate: $successCount/$($texts.Count) files generated" -ForegroundColor Green

# List all generated files
Write-Host "`nGenerated files:" -ForegroundColor Cyan
$outputFiles = Get-ChildItem "$outputDir\*.wav" | Sort-Object Name
if ($outputFiles.Count -gt 0) {
  foreach ($file in $outputFiles) {
    $sizeKB = [Math]::Round($file.Length / 1024, 1)
    Write-Host "  $($file.Name) - $sizeKB KB" -ForegroundColor White
  }
}
else {
  Write-Host "  No audio files found in output directory" -ForegroundColor Red
}

Write-Host "`nOutput directory: $outputDir" -ForegroundColor Gray