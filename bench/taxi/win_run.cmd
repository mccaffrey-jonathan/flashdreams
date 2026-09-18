@echo off
REM Windows-native Crazy Robotaxi launcher: MSVC build env (native FP8 DiT / PhysX builds) + uv venv.
REM usage: win_run.cmd SLUG [run-v2 args...] -- [app args...]
REM env:   FLASHDREAMS_ROOT (repo checkout, default: two levels above this script)
REM        VCVARS64         (vcvars64.bat, default: VS 2022 BuildTools)
REM        HF_HOME          keep it SHORT: omni-dreams-samples file paths exceed MAX_PATH under a
REM                         long cache root unless LongPathsEnabled is set (junctions do not help,
REM                         huggingface_hub resolves them)
if "%FLASHDREAMS_ROOT%"=="" set FLASHDREAMS_ROOT=%~dp0..\..
if "%VCVARS64%"=="" set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS64%" >nul
cd /d "%FLASHDREAMS_ROOT%"
python -m uv run --no-sync flashdreams-run-v2 %*
