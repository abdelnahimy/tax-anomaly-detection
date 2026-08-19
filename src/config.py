"""Configuration centrale du projet : chemins, constantes, listes de variables.

Toute la logique de résolution de chemins vit ici pour que les notebooks
n'aient plus besoin de chemins relatifs codés en dur (ex. ``"../data/raw/..."``),
qui cassent dès que le notebook est lancé depuis un autre répertoire de travail
(VS Code, JupyterLab, `jupyter nbconvert --execute`, CI, etc.).
"""
from pathlib import Path


def find_project_root(marker: str = "src") -> Path:
    """Localise la racine du projet (le dossier contenant ``marker``, par défaut ``src/``).

    Deux stratégies, dans l'ordre :

    1. remonter depuis le **répertoire de travail courant** — couvre le cas
       normal d'un notebook lancé depuis ``notebooks/`` ou depuis la racine ;
    2. à défaut, remonter depuis **l'emplacement de ce fichier** — indispensable
       quand le processus est lancé depuis un tout autre répertoire, par exemple
       ``streamlit run /chemin/absolu/tax-anomaly-detection/dashboard/app.py``
       exécuté depuis le dossier personnel de l'utilisateur. Sans ce repli, le
       dashboard plantait au démarrage selon le répertoire courant.

    Raises
    ------
    FileNotFoundError
        Si aucune des deux stratégies n'aboutit.
    """
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parent]

    for start in candidates:
        for parent in [start, *start.parents]:
            if (parent / marker).is_dir():
                return parent

    raise FileNotFoundError(
        f"Impossible de localiser la racine du projet (dossier '{marker}/' introuvable "
        f"en remontant depuis {candidates[0]} ni depuis {candidates[1]}). "
        f"Vérifiez que le dossier src/ est bien présent dans le projet."
    )


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / "data_FraudDetection_JAR2020.csv"
DATA_PROCESSED = DATA_DIR / "processed" / "data_transformed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _dir in (DATA_DIR / "processed", MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Reproductibilité ------------------------------------------------------
RANDOM_STATE = 42

# --- Variables du dataset JAR2020 (Bao, Ke, Li, Yu & Zhang, 2020) ----------
ID_VARS = ["fyear", "gvkey", "p_aaer", "misstate"]

RAW_ACCOUNTING_VARS = [
    "act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
    "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
    "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f",
]

RATIO_VARS = [
    "dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets", "ch_cs",
    "ch_cm", "ch_roa", "issue", "bm", "dpi", "reoa", "EBIT", "ch_fcf",
]

# --- Protocole de validation temporelle ------------------------------------
# Répliqué de run_RUSBoost.m (Bao et al., 2020) : test sur 2003-2014, gap de
# 2 ans entre fin du train et année de test.
TEST_YEAR_START = 2003
TEST_YEAR_END = 2014
TEST_YEARS = list(range(TEST_YEAR_START, TEST_YEAR_END + 1))
SPLIT_GAP_YEARS = 2
DATA_YEAR_START = 1991

# Seuils utilisés dans plusieurs notebooks
STRONG_CORRELATION_THRESHOLD = 0.85
CONTAMINATION = 0.0066  # taux de fraude empirique (cf. 01_EDA), utilisé par IF/LOF/AE
