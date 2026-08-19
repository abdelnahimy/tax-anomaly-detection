"""Charte graphique du tableau de bord — registre administratif neutre.

Parti pris
----------
Un outil destiné à une brigade de vérification n'a pas à séduire : il a à être
**lu vite et sans ambiguïté**, y compris imprimé en noir et blanc et projeté en
salle de soutenance. La charte est donc quasi monochrome — une échelle de gris
ardoise — avec **une seule couleur d'accent**, employée uniquement là où elle
porte une information.

Ce qui en découle, et qui n'est pas négociable :

- **aucun dégradé, aucune pastille colorée, aucun aplat de couleur vive** ;
  la hiérarchie vient de la typographie, du filet et du blanc ;
- **aucun emoji** — ni dans les titres, ni dans les libellés d'onglets, ni dans
  les tableaux ;
- les couleurs de statut (risque élevé / moyen / faible) sont **réservées** à
  cet usage et ne servent jamais de couleur de série dans un graphique ;
- elles s'accompagnent **toujours d'un libellé texte**, jamais de la couleur
  seule — c'est la condition pour rester lisible en vision déficiente et en
  impression monochrome.

Palette de graphiques
---------------------
Deux teintes catégorielles suffisent : aucun graphique du tableau de bord ne
compare plus de deux séries.

    accent   #2C5F9E   bleu acier    — série principale, valeur mise en avant
    serie_2  #A05B1E   ocre brûlée   — série de comparaison, pôle opposé

Ce couple a été validé, pas choisi à l'œil : bande de clarté, plancher de
chroma, séparation en vision déficiente (ΔE 19,7 protan · 22,1 tritan) et
contraste sur fond clair sont tous conformes. Il sert aussi de paire divergente
(froid ↔ chaud) pour la décomposition SHAP, avec un gris neutre au point milieu.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    # Surfaces
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "surface_creuse": "#F1F5F9",
    # Filets
    "filet": "#E2E8F0",
    "filet_fort": "#CBD5E1",
    # Encres
    "encre": "#0F172A",
    "encre_moyenne": "#475569",
    "encre_faible": "#94A3B8",
    # Accent unique
    "accent": "#2C5F9E",
    "accent_fonce": "#1F4470",
    "accent_pale": "#EEF3F9",
    # Deuxième teinte catégorielle / pôle chaud de la paire divergente
    "serie_2": "#A05B1E",
    # Statuts — réservés, toujours accompagnés d'un libellé
    "risque_eleve": "#A03530",
    "risque_moyen": "#8A6420",
    "risque_faible": "#4A7059",
}

# Alias courts, pour les appels matplotlib
SURFACE = PALETTE["surface"]
FILET = PALETTE["filet"]
ENCRE = PALETTE["encre"]
ENCRE_MOYENNE = PALETTE["encre_moyenne"]
ENCRE_FAIBLE = PALETTE["encre_faible"]
ACCENT = PALETTE["accent"]
ACCENT_FONCE = PALETTE["accent_fonce"]
SERIE_2 = PALETTE["serie_2"]


# ---------------------------------------------------------------------------
# Feuille de style
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
:root {{
  --surface:        {PALETTE["surface"]};
  --surface-alt:    {PALETTE["surface_alt"]};
  --surface-creuse: {PALETTE["surface_creuse"]};
  --filet:          {PALETTE["filet"]};
  --filet-fort:     {PALETTE["filet_fort"]};
  --encre:          {PALETTE["encre"]};
  --encre-moyenne:  {PALETTE["encre_moyenne"]};
  --encre-faible:   {PALETTE["encre_faible"]};
  --accent:         {PALETTE["accent"]};
  --accent-fonce:   {PALETTE["accent_fonce"]};
  --accent-pale:    {PALETTE["accent_pale"]};
  --serie-2:        {PALETTE["serie_2"]};
}}

/* --- Fond et largeur de colonne --------------------------------------- */
[data-testid="stAppViewContainer"] {{ background: var(--surface-alt); }}
[data-testid="stAppViewContainer"] .block-container {{
  padding-top: 2.2rem;
  padding-bottom: 4rem;
  max-width: 1280px;
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* --- Typographie -------------------------------------------------------- */
html, body, [data-testid="stAppViewContainer"] {{
  color: var(--encre);
  -webkit-font-smoothing: antialiased;
}}
h1, h2, h3, h4 {{ color: var(--encre); letter-spacing: -0.01em; }}
h2 {{ font-size: 1.30rem; font-weight: 600; }}
h3 {{ font-size: 1.06rem; font-weight: 600; }}
[data-testid="stCaptionContainer"] p {{ color: var(--encre-moyenne) !important; }}

/* --- En-tête ------------------------------------------------------------ */
.entete {{
  border-bottom: 1px solid var(--filet);
  padding-bottom: 1.1rem;
  margin-bottom: 1.6rem;
}}
.entete .mention {{
  font-size: 0.70rem;
  font-weight: 600;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--encre-faible);
  margin-bottom: 0.5rem;
}}
.entete h1 {{
  font-size: 1.60rem;
  font-weight: 600;
  margin: 0 0 0.35rem 0;
  line-height: 1.25;
  padding: 0;
}}
.entete .sous-titre {{
  font-size: 0.93rem;
  color: var(--encre-moyenne);
  max-width: 64ch;
  line-height: 1.5;
  margin: 0;
}}

/* --- Titre d'étape ------------------------------------------------------ */
.etape {{ margin: 0 0 0.9rem 0; }}
.etape .rang {{
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  display: block;
  margin-bottom: 0.22rem;
}}
.etape .intitule {{
  font-size: 1.12rem;
  font-weight: 600;
  color: var(--encre);
  line-height: 1.3;
}}

/* --- Note encadrée (remplace les bandeaux colorés) ---------------------- */
.note {{
  border: 1px solid var(--filet);
  border-left: 3px solid var(--filet-fort);
  background: var(--surface);
  padding: 0.78rem 1rem;
  margin: 0.55rem 0 1rem 0;
  font-size: 0.885rem;
  line-height: 1.55;
  color: var(--encre-moyenne);
}}
.note b, .note strong {{ color: var(--encre); font-weight: 600; }}
.note code {{
  background: var(--surface-creuse);
  padding: 0.08rem 0.32rem;
  font-size: 0.85em;
  color: var(--encre);
}}
.note-reference {{ border-left-color: var(--accent); }}
.note-attention {{ border-left-color: var(--serie-2); }}
.note-alerte    {{ border-left-color: {PALETTE["risque_eleve"]}; }}

/* --- Onglets ------------------------------------------------------------ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 0;
  border-bottom: 1px solid var(--filet);
  margin-bottom: 1.7rem;
}}
[data-testid="stTabs"] button[data-baseweb="tab"] {{
  background: transparent;
  padding: 0.6rem 1.15rem;
  font-size: 0.92rem;
  font-weight: 500;
  color: var(--encre-moyenne);
}}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--encre);
  font-weight: 600;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
  background-color: var(--accent);
  height: 2px;
}}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

/* --- Indicateurs chiffrés ----------------------------------------------- */
[data-testid="stMetric"] {{
  background: var(--surface);
  border: 1px solid var(--filet);
  padding: 0.8rem 0.95rem;
}}
[data-testid="stMetricLabel"] p {{
  font-size: 0.72rem !important;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--encre-faible) !important;
}}
[data-testid="stMetricValue"] {{
  font-size: 1.40rem;
  font-weight: 600;
  color: var(--encre);
}}

/* --- Boutons ------------------------------------------------------------- */
.stButton > button, [data-testid="stDownloadButton"] > button {{
  background: var(--surface);
  color: var(--encre);
  border: 1px solid var(--filet-fort);
  border-radius: 2px;
  font-weight: 500;
  font-size: 0.88rem;
  padding: 0.42rem 0.95rem;
  transition: none;
}}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {{
  border-color: var(--accent);
  color: var(--accent);
  background: var(--surface);
}}
.stButton > button:focus, [data-testid="stDownloadButton"] > button:focus {{
  box-shadow: none;
  border-color: var(--accent);
  color: var(--accent);
}}

/* --- Tableaux ------------------------------------------------------------ */
[data-testid="stDataFrame"] {{ border: 1px solid var(--filet); border-radius: 0; }}

/* --- Champs de saisie ---------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {{
  background: var(--surface);
  border: 1px dashed var(--filet-fort);
  border-radius: 2px;
}}
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
  border-radius: 2px;
}}

/* --- Volets dépliants ----------------------------------------------------- */
[data-testid="stExpander"] {{
  border: 1px solid var(--filet);
  border-radius: 0;
  background: var(--surface);
}}
[data-testid="stExpander"] summary {{
  font-size: 0.87rem;
  font-weight: 500;
  color: var(--encre-moyenne);
}}

/* --- Barre latérale -------------------------------------------------------- */
[data-testid="stSidebar"] {{
  background: var(--surface);
  border-right: 1px solid var(--filet);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  font-size: 0.71rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--encre-faible);
  margin-bottom: 0.6rem;
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {{
  font-size: 0.855rem;
  line-height: 1.55;
  color: var(--encre-moyenne);
}}
[data-testid="stSidebar"] strong {{ color: var(--encre); }}
[data-testid="stSidebar"] hr {{ margin: 1.1rem 0; border-color: var(--filet); }}

/* --- Divers ---------------------------------------------------------------- */
hr {{ border-color: var(--filet); }}
[data-testid="stSlider"] [role="slider"] {{ border-color: var(--accent); }}
</style>
"""


def injecter_css() -> None:
    """Applique la feuille de style. À appeler une fois, après `set_page_config`."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Composants
# ---------------------------------------------------------------------------
def entete(titre: str, sous_titre: str, mention: str | None = None) -> None:
    """En-tête de page. Remplace `st.title`, sans aplat ni dégradé."""
    bloc_mention = f'<div class="mention">{mention}</div>' if mention else ""
    st.markdown(
        f'<div class="entete">{bloc_mention}'
        f"<h1>{titre}</h1>"
        f'<p class="sous-titre">{sous_titre}</p></div>',
        unsafe_allow_html=True,
    )


def etape(numero: int, titre: str) -> None:
    """Titre d'étape numérotée, sans pastille colorée."""
    st.markdown(
        f'<div class="etape"><span class="rang">Étape {numero}</span>'
        f'<span class="intitule">{titre}</span></div>',
        unsafe_allow_html=True,
    )


def note(texte: str, ton: str = "neutre") -> None:
    """Encadré informatif.

    ``ton`` vaut ``"neutre"``, ``"reference"``, ``"attention"`` ou ``"alerte"``.
    Il ne change que la couleur du filet de gauche, jamais le fond. Le texte
    accepte du HTML en ligne (``<b>``, ``<code>``).
    """
    classe = {
        "neutre": "note",
        "reference": "note note-reference",
        "attention": "note note-attention",
        "alerte": "note note-alerte",
    }.get(ton, "note")
    st.markdown(f'<div class="{classe}">{texte}</div>', unsafe_allow_html=True)


def libelle_risque(percentile: float) -> str:
    """Libellé textuel de la bande de risque.

    Volontairement sans couleur ni pictogramme : le rang est l'information, et
    il doit rester lisible en impression monochrome.
    """
    if percentile >= 99:
        return "Premier centile — priorité haute"
    if percentile >= 95:
        return "Cinq premiers pour cent — à examiner"
    if percentile >= 90:
        return "Dix premiers pour cent — surveillance"
    return "Reste du portefeuille"


def style_axes(ax) -> None:
    """Applique le registre neutre à un axe matplotlib.

    Filets discrets, pas de cadre, encre secondaire pour les graduations : la
    donnée doit être ce qu'on voit en premier.
    """
    ax.set_facecolor(SURFACE)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor(SURFACE)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(FILET)
        ax.spines[cote].set_linewidth(1.0)
    ax.tick_params(colors=ENCRE_MOYENNE, labelsize=9, length=0)
    ax.xaxis.label.set_color(ENCRE_MOYENNE)
    ax.yaxis.label.set_color(ENCRE_MOYENNE)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.grid(axis="x", color=FILET, linewidth=0.8)
    ax.set_axisbelow(True)
