@echo off
setlocal
where python >nul 2>&1
if %errorlevel%==0 (
  python -m pip install -r requirements.txt
) else (
  "%LocalAppData%\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 exit /b 1
echo Setup complete. Start the app with: python run.py
endlocal
