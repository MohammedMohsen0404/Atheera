Add-Type -AssemblyName System.Speech
$lines = Get-Content -Raw "D:\Atheer\video\narration.json" | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Zira Desktop"); $s.Rate = 0
for ($i=0; $i -lt $lines.Count; $i++) { $s.SetOutputToWaveFile("D:\Atheer\video\audio\s$i.wav"); $s.Speak($lines[$i]) }
$s.SetOutputToNull()
