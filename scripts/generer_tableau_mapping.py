"""Régénère la section « tableau de correspondance » de 02_mapping_liasse_fiscale.md.

Le tableau du document est produit à partir de `src/liasse_mapping.LIASSE_MAPPING`,
et non recopié à la main : une correspondance modifiée dans le code se répercute
donc dans la documentation, et les deux ne peuvent pas diverger silencieusement.

Usage :
    python scripts/generer_tableau_mapping.py          # réécrit le bloc du .md
    python scripts/generer_tableau_mapping.py --csv    # exporte aussi reports/mapping_liasse_fiscale.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.liasse_mapping import LIASSE_MAPPING, table_correspondance  # noqa: E402
from src.config import REPORTS_DIR  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "docs" / "02_mapping_liasse_fiscale.md"
BALISE_DEBUT = "<!-- DEBUT TABLEAU GENERE -->"
BALISE_FIN = "<!-- FIN TABLEAU GENERE -->"

SYMBOLE_FIDELITE = {
    "directe": "🟢 directe",
    "agregat": "🔵 agrégat",
    "approximation": "🟠 approximation",
    "indisponible": "⚪ indisponible",
}


def bloc_markdown() -> str:
    lignes = [
        "| # | Variable | Définition Compustat | Poste de la liasse marocaine | Tableau | Comptes CGNC | Fidélité |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(LIASSE_MAPPING, start=1):
        lignes.append(
            f"| {i} | `{c.variable}` | {c.libelle_us} | {c.poste_liasse} | "
            f"{c.tableau} | {c.comptes_cgnc} | {SYMBOLE_FIDELITE.get(c.fidelite, c.fidelite)} |"
        )

    lignes.append("")
    lignes.append("### Notes de correspondance, variable par variable")
    lignes.append("")
    for c in LIASSE_MAPPING:
        lignes.append(f"**`{c.variable}` — {c.poste_liasse}**  ")
        lignes.append(f"*{SYMBOLE_FIDELITE.get(c.fidelite, c.fidelite)}* — {c.remarque}")
        lignes.append("")
    return "\n".join(lignes)


def main() -> None:
    texte = DOC.read_text(encoding="utf-8")
    if BALISE_DEBUT not in texte or BALISE_FIN not in texte:
        raise SystemExit(
            f"Balises {BALISE_DEBUT} / {BALISE_FIN} introuvables dans {DOC.name}."
        )
    avant = texte.split(BALISE_DEBUT)[0]
    apres = texte.split(BALISE_FIN)[1]
    DOC.write_text(
        f"{avant}{BALISE_DEBUT}\n\n{bloc_markdown()}\n{BALISE_FIN}{apres}",
        encoding="utf-8",
    )
    print(f"Tableau régénéré dans {DOC} ({len(LIASSE_MAPPING)} variables).")

    if "--csv" in sys.argv:
        cible = Path(REPORTS_DIR) / "mapping_liasse_fiscale.csv"
        table_correspondance().to_csv(cible, index=False, encoding="utf-8-sig")
        print(f"Export CSV : {cible}")


if __name__ == "__main__":
    main()
