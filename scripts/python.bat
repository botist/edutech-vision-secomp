@echo off
setlocal
set "ROOT=%~dp0.."
set "PYTHON="

if exist "%ROOT%\.venv\Scripts\python.exe" set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%\.venv\python.exe" set "PYTHON=%ROOT%\.venv\python.exe"

if not defined PYTHON (
  echo Python local nao encontrado. Rode run.bat primeiro.
  exit /b 1
)

"%PYTHON%" %*
exit /b %ERRORLEVEL%
