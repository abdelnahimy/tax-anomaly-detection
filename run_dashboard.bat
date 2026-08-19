@echo off
REM ---------------------------------------------------------------------------
REM Lancement du dashboard de detection d'anomalies fiscales (Windows).
REM
REM Utilise en priorite l'environnement virtuel .venv s'il existe (cree par
REM setup_venv.bat), sinon le Python du systeme.
REM
REM Appelle toujours "python -m streamlit" plutot que "streamlit" : quand pip
REM installe en mode utilisateur, le dossier Scripts\ n'est pas dans le PATH et
REM la commande "streamlit" reste introuvable meme si le paquet est installe.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

echo.
echo === Ciblage du controle fiscal - DRI Tetouan ===
echo.

REM --- 1. Choix de l'interpreteur ------------------------------------------
if exist ".venv\Scripts\python.exe" (
    set PYEXE=.venv\Scripts\python.exe
    echo Environnement : .venv
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] Python est introuvable dans le PATH.
        echo Lancez d'abord setup_venv.bat, ou installez Python depuis python.org
        echo en cochant "Add Python to PATH".
        echo.
        pause
        exit /b 1
    )
    set PYEXE=python
    echo Environnement : Python du systeme ^(aucun .venv trouve^)
)

REM --- 2. Streamlit est-il disponible ? ------------------------------------
"%PYEXE%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERREUR] Streamlit n'est pas installe dans cet environnement.
    echo Lancez setup_venv.bat pour installer les dependances.
    echo.
    pause
    exit /b 1
)

REM --- 3. Le modele JAR2020 est-il present ? -------------------------------
REM Non bloquant : aucun des trois ecrans n'en depend. Seule une ligne
REM descriptive de l'ecran "Methode et resultats" est remplacee par une note.
if not exist "models\xgboost_v2_best.pkl" (
    echo.
    echo [INFO] models\xgboost_v2_best.pkl est introuvable.
    echo Le tableau de bord fonctionne quand meme, dans son integralite.
    echo Pour regenerer le modele, executez le notebook
    echo 05_modeling_supervised_v2.ipynb.
    echo.
)

REM --- 4. Lancement --------------------------------------------------------
echo.
echo Demarrage du dashboard... ^(Ctrl+C pour arreter^)
echo Le navigateur s'ouvre sur http://localhost:8501
echo Trois ecrans : "Ciblage" pour produire la liste de travail,
echo "Analyse d'un dossier" pour le detail, "Methode et resultats"
echo pour la soutenance.
echo.
"%PYEXE%" -m streamlit run dashboard/app.py

pause
