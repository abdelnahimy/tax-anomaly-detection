@echo off
REM ---------------------------------------------------------------------------
REM Nettoyage du dossier de projet (Windows).
REM
REM Supprime ce qui se regenere tout seul (caches Python, caches pytest) et les
REM fichiers devenus obsoletes. Ne touche JAMAIS au code, a la documentation,
REM aux notebooks, aux modeles entrainees ni aux donnees.
REM
REM Sans danger : tout ce qui est supprime ici est soit un cache, soit un
REM fichier explicitement remplace ailleurs dans le projet.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

echo.
echo === Nettoyage du projet ===
echo.

REM --- 1. Caches Python et pytest ------------------------------------------
REM Regeneres automatiquement a la prochaine execution. Deja ignores par git.
echo [1/4] Caches Python et pytest...
for /d /r %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d"
)
for /d /r %%d in (.ipynb_checkpoints) do (
    if exist "%%d" rd /s /q "%%d"
)
if exist ".pytest_cache" rd /s /q ".pytest_cache"
if exist ".mypy_cache"   rd /s /q ".mypy_cache"
if exist ".ruff_cache"   rd /s /q ".ruff_cache"
echo       fait.

REM --- 2. Archives de synchronisation ---------------------------------------
REM Copies temporaires deposees pendant les mises a jour du projet.
echo [2/4] Archives de synchronisation...
if exist "_to_delete" (
    rd /s /q "_to_delete"
    echo       _to_delete\ supprime.
) else (
    echo       rien a supprimer.
)

REM --- 3. Fichiers remplaces -------------------------------------------------
REM Tout ce qui est supprime ici a ete fusionne ou deplace ailleurs :
REM   - app_technique.py         -> fusionne dans dashboard\app.py
REM   - generer_tableau_mapping  -> deplace dans scripts\
REM   - requirements-*.txt       -> fusionnes dans requirements.txt
REM   - README.md secondaires    -> fusionnes dans le README.md racine
echo [3/4] Fichiers remplaces...
if exist "dashboard\app_technique.py" (
    del /q "dashboard\app_technique.py"
    echo       dashboard\app_technique.py supprime ^(fusionne dans app.py^).
)
if exist "docs\generer_tableau_mapping.py" (
    del /q "docs\generer_tableau_mapping.py"
    echo       docs\generer_tableau_mapping.py supprime ^(deplace dans scripts\^).
)
for %%f in (requirements-dashboard.txt requirements-dev.txt requirements-optional-torch.txt) do (
    if exist "%%f" (
        del /q "%%f"
        echo       %%f supprime ^(fusionne dans requirements.txt^).
    )
)
for %%f in ("dashboard\README.md" "docs\README.md" "data\raw\README.md") do (
    if exist %%f (
        del /q %%f
        echo       %%~f supprime ^(fusionne dans le README.md racine^).
    )
)
if exist "data\external" (
    rd /q "data\external" 2>nul
    if not exist "data\external" echo       data\external\ supprime ^(dossier vide^).
)
echo       fait.

REM --- 4. Etat du dossier ----------------------------------------------------
echo [4/4] Etat du dossier :
echo.
if exist "data\processed\data_transformed.csv" (
    echo       data\processed\data_transformed.csv    ~114 Mo
    echo         Fichier intermediaire, regenerable par le notebook 02.
    echo         Necessaire aux notebooks 03, 04 et 05. NON supprime :
    echo         effacez-le a la main si vous avez besoin de la place.
)
if exist "data\raw\data_FraudDetection_JAR2020.csv" (
    echo       data\raw\data_FraudDetection_JAR2020.csv  ~46 Mo
    echo         Jeu source JAR2020. NON supprime : les notebooks 01 a 05
    echo         en dependent. Voir data\raw\README.md pour le retelecharger.
)
if exist ".venv" (
    echo       .venv\
    echo         Environnement virtuel. NON supprime : l'application en a
    echo         besoin. Recreable par setup_venv.bat.
)
echo.
echo === Termine ===
echo.
pause
