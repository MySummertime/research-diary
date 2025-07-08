@echo off
REM This script activates the Conda environment and starts the Uvicorn server for the Risk_Assessment_Models project.

REM --- 1. Set Project Root and PYTHONPATH ---
REM Get the directory where this script is located (i.e., the project root).
REM %~dp0 gives the directory of this script, which is Risk_Assessment_Models.
set "PROJECT_ROOT=%~dp0"

REM Add the project root to the PYTHONPATH. This allows Python to find modules
REM starting from the 'backend' directory (e.g., 'from backend.api import main').
set "PYTHONPATH=%PROJECT_ROOT%"
echo PYTHONPATH set to: %PROJECT_ROOT%

REM --- 2. Activate Conda Environment ---
REM Use 'python.exe' directly from the system's PATH if available,
REM avoiding a hardcoded path to the base installation.
echo Activating Conda environment...

REM Define the full path to the project's specific environment.
set "CONDA_ENV_PATH=%PROJECT_ROOT%backend\envs"

REM Check if the conda environment directory exists.
if not exist "%CONDA_ENV_PATH%" (
    echo.
    echo ERROR: Conda environment not found at the expected location:
    echo %CONDA_ENV_PATH%
    echo Please create it first using the 'environment.yaml' file.
    echo.
    pause
    exit /b 1
)

REM Activate the conda environment by its full path.
REM The 'call' command ensures that the script continues after conda is done.
call conda activate "%CONDA_ENV_PATH%"

REM Check if the activation was successful by verifying the CONDA_PREFIX variable.
if "%CONDA_PREFIX%" neq "%CONDA_ENV_PATH%" (
    echo.
    echo ERROR: Failed to activate the Conda environment.
    echo Please ensure Conda is installed and accessible from your terminal.
    echo Try running 'conda init' in your terminal first.
    echo.
    pause
    exit /b 1
)

echo Conda environment successfully activated: %CONDA_PREFIX%
echo.

REM --- 3. Run Uvicorn Server ---
echo Starting FastAPI server with Uvicorn...
echo Visit http://localhost:8000 in your browser.
echo Press CTRL+C to stop the server.
echo.

REM The import path 'backend.api.main:app' works because PYTHONPATH is set correctly.
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

REM Add a pause at the end so the user can see any final messages before the window closes.
echo.
echo Server has been stopped.
pause

REM End of script
exit /b 0