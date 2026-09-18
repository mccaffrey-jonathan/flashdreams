# Screen-capture the native FlashDreams game window while scripted key presses drive the taxi.
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File win_drive_capture.ps1 -Out E:\path\drive.mp4 [-Seconds 90]
param(
    [string]$Out = "E:\flashdreams\bench\taxi\runs\win_fast_window\drive.mp4",
    [int]$Seconds = 90,
    [int]$WaitForWindowSeconds = 300,
    [int]$WarmupSeconds = 45
)
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Key {
    [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    public const uint KEYUP = 0x2;
    public static void Down(byte vk) { keybd_event(vk, 0, 0, UIntPtr.Zero); }
    public static void Up(byte vk) { keybd_event(vk, 0, KEYUP, UIntPtr.Zero); }
}
"@
$VK_W = 0x57; $VK_A = 0x41; $VK_D = 0x44

$deadline = (Get-Date).AddSeconds($WaitForWindowSeconds)
$h = [IntPtr]::Zero
while ((Get-Date) -lt $deadline) {
    $proc = Get-Process | Where-Object { $_.MainWindowTitle -eq "FlashDreams" } | Select-Object -First 1
    if ($proc -and $proc.MainWindowHandle -ne 0) { $h = $proc.MainWindowHandle; break }
    Start-Sleep -Seconds 2
}
if ($h -eq [IntPtr]::Zero) { Write-Error "FlashDreams window not found"; exit 1 }
Write-Host "window $h found; waiting $WarmupSeconds s for gameplay to start"
Start-Sleep -Seconds $WarmupSeconds
[Key]::SetForegroundWindow($h) | Out-Null
Start-Sleep -Milliseconds 500

$ff = Start-Process -FilePath ffmpeg -ArgumentList @("-y", "-f", "gdigrab", "-framerate", "30", "-i", "title=FlashDreams", "-t", "$Seconds", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "`"$Out`"") -PassThru -WindowStyle Hidden
Write-Host "ffmpeg pid $($ff.Id) recording $Seconds s"

# drive pattern: accelerate, weave, accelerate. Keys are held with keybd_event so the game sees real key-down/up.
$plan = @(
    @{ keys = @($VK_W);        s = 20 },
    @{ keys = @($VK_W, $VK_A); s = 3 },
    @{ keys = @($VK_W);        s = 12 },
    @{ keys = @($VK_W, $VK_D); s = 3 },
    @{ keys = @($VK_W);        s = 20 },
    @{ keys = @($VK_W, $VK_A); s = 2 },
    @{ keys = @($VK_W);        s = 25 }
)
foreach ($step in $plan) {
    [Key]::SetForegroundWindow($h) | Out-Null
    foreach ($k in $step.keys) { [Key]::Down([byte]$k) }
    Start-Sleep -Seconds $step.s
    foreach ($k in $step.keys) { [Key]::Up([byte]$k) }
    if ($ff.HasExited) { break }
}
$ff.WaitForExit()
Write-Host "capture done: $Out"
