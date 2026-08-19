@echo off
setlocal enabledelayedexpansion
REM ---------------------------------------------------------------------------
REM Cree un environnement virtuel Python isole (.venv) pour le projet et y
REM installe les dependances.
REM
REM Concu pour resister aux erreurs "ReadTimeoutError" de pip sur connexion lente :
REM   - timeout de lecture porte a 120s (defaut : 15s)
REM   - 10 tentatives par paquet
REM   - installation paquet par paquet : un echec ne perd pas tout le reste,
REM     et le cache pip conserve ce qui a deja ete telecharge
REM
REM Messages en ASCII pur (les accents s'affichent mal dans la console Windows).
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

echo.
echo ==========================================================
echo   Detection d'anomalies fiscales - installation
echo ==========================================================
echo.

REM --- 1. Choix de l'interpreteur Python -----------------------------------
REM Python 3.11 est privilegie : toutes les dependances y disposent de roues
REM precompilees pour Windows (verifie). Les versions tres recentes de Python
REM n'ont pas toujours de roue disponible et declenchent des compilations locales.
set PYEXE=
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYEXE=py -3.11
    echo Interpreteur : Python 3.11 ^(recommande^)
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] Aucun Python trouve. Installez Python 3.11 depuis python.org
        echo en cochant "Add Python to PATH", puis relancez ce script.
        echo.
        pause
        exit /b 1
    )
    set PYEXE=python
    for /f "delims=" %%v in ('python --version 2^>^&1') do echo Interpreteur : %%v
    echo [INFO] Python 3.11 non detecte, utilisation du Python par defaut.
)

REM --- 2. Creation du venv --------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo Environnement .venv deja present, reutilisation.
) else (
    echo Creation de l'environnement virtuel .venv ...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo [ERREUR] Echec de la creation du venv.
        pause
        exit /b 1
    )
)

set VPY=.venv\Scripts\python.exe

REM --- 3. Mise a jour de pip ------------------------------------------------
echo.
echo Mise a jour de pip ...
"%VPY%" -m pip install --upgrade pip --timeout 120 --retries 10 --quiet

REM --- 4. Installation paquet par paquet ------------------------------------
REM Installer ligne par ligne evite qu'un seul timeout annule toute la session.
echo.
echo Installation des dependances ^(patience, cela peut prendre plusieurs minutes^)...
echo.

set FAILED=
for /f "usebackq eol=# tokens=* delims=" %%p in ("requirements.txt") do (
    set LINE=%%p
    if not "!LINE!"=="" (
        echo   -^> !LINE!
        "%VPY%" -m pip install "!LINE!" --timeout 120 --retries 10 --prefer-binary --quiet
        if errorlevel 1 (
            echo      [ECHEC] !LINE!
            set FAILED=!FAILED! !LINE!
        )
    )
)

REM --- 5. Verification ------------------------------------------------------
echo.
echo ==========================================================
if not "!FAILED!"=="" (
    echo [ATTENTION] Paquets non installes :!FAILED!
    echo Relancez simplement ce script : pip reprendra ou il s'est arrete.
    echo.
)

REM scipy est verifie en plus : il n'est pas importe par le dashboard mais l'est
REM par scripts\robustesse_variables_absentes.py (correlations de rang).
"%VPY%" -c "import streamlit, pandas, numpy, scipy, sklearn, xgboost, shap, matplotlib, seaborn, joblib; print('Verification des imports : OK')"
if errorlevel 1 (
    echo [ERREUR] Certaines dependances manquent encore. Relancez ce script.
    echo.
    pause
    exit /b 1
)

echo.
echo Installation terminee.
echo.
echo Pour lancer le dashboard :
echo     run_dashboard.bat
echo.
echo Pour utiliser cet environnement dans Jupyter / VS Code, selectionnez
echo l'interpreteur : .venv\Scripts\python.exe
echo.
echo Pour executer la suite de tests :
echo     .venv\Scripts\python.exe -m pytest
echo.
pause
