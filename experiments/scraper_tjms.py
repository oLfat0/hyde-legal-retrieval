"""
TJMS eSAJ Scraper
==================
Performs 3 searches on https://esaj.tjms.jus.br/cjsg/resultadoCompleta.do:
  - Apelação Cível
  - Embargos de Declaração
  - Apelação Criminal

Collects up to LIMIT_PER_SEARCH results per search term, deduplicating by
numero_processo across all searches. The 'classe' field is extracted directly
from the "Classe/Assunto" row in each result card on the page.

Output: ementas.json with fields: numero_processo, cdacordao, classe, texto.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python scraper_tjms.py
    python scraper_tjms.py --paginas 3       # pages per search term
    python scraper_tjms.py --headless false  # show browser window
"""

import json
import time
import argparse
from collections import Counter
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# ── Settings ───────────────────────────────────────────────────────────────────
BASE_URL          = "https://esaj.tjms.jus.br/cjsg/resultadoCompleta.do"
OUTPUT_FILE       = "ementas.json"
DELAY_S           = 0.8   # pause between requests (seconds)
LIMIT_PER_SEARCH  = 50    # max ementas per search term

SEARCH_TERMS = [
    "Apelação Cível",
    "Embargos de Declaração",
    "Apelação Criminal",
]
# ──────────────────────────────────────────────────────────────────────────────


def collect_page_metadata(page) -> list[dict]:
    """
    Extracts metadata for every result card on the current page via JS.
    Returns a list of dicts with: cdacordao, numero_processo, classe.

    The 'classe' field comes from the 'Classe/Assunto' cell in the result row
    (tr.ementaClass / tr.ementaClass2), which is the closest sibling of the
    download link for that record.
    """
    return page.evaluate("""
        () => {
            const out = [];
            document.querySelectorAll("a[onclick*='abrirPopUpDadosSemFormatacao']")
                .forEach(btn => {
                    const m = (btn.getAttribute("onclick") || "")
                              .match(/abrirPopUpDadosSemFormatacao\\((\\d+)\\)/);
                    if (!m) return;
                    const cd = m[1];

                    // numero_processo from the download link
                    const dl = document.querySelector(
                        "a.downloadEmenta[cdacordao='" + cd + "']"
                    );
                    const numero = dl ? dl.innerText.trim() : cd;

                    // classe from the nearest "Classe/Assunto:" table cell
                    // Each result is wrapped in a <table>; walk up to find it
                    let classe = "";
                    let node = btn.closest("table");
                    if (node) {
                        const rows = node.querySelectorAll("tr");
                        for (const row of rows) {
                            const strong = row.querySelector("strong");
                            if (strong && strong.innerText.includes("Classe/Assunto")) {
                                // Text after the <strong> tag
                                const raw = row.querySelector("td").innerText;
                                const parts = raw.split(":");
                                if (parts.length >= 2) {
                                    classe = parts.slice(1).join(":").trim();
                                }
                                break;
                            }
                        }
                    }

                    out.push({ cdacordao: cd, numero_processo: numero, classe });
                });
            return out;
        }
    """)


def collect_ementas_from_page(
    page,
    restantes: int,
    seen: set,
) -> list[dict]:
    """
    Collects ementas from the current results page.

    Args:
        page:      Playwright page object.
        restantes: How many more ementas are allowed before hitting the limit.
        seen:      Set of already-collected numero_processo values (mutated in place).

    Returns:
        List of ementa dicts: numero_processo, cdacordao, classe, texto.
    """
    results = []
    records = collect_page_metadata(page)
    print(f"  → {len(records)} result(s) on page")

    for rec in records:
        if restantes <= 0:
            break

        cdacordao       = rec["cdacordao"]
        numero_processo = rec["numero_processo"]
        classe          = rec["classe"]
        div_id          = f"textAreaDados_{cdacordao}"

        # Skip duplicates found across different search terms
        if numero_processo in seen:
            print(f"    ↷ {numero_processo} already collected — skipping")
            continue

        try:
            # Trigger the site's own JS function to inject the text div
            page.evaluate(f"abrirPopUpDadosSemFormatacao({cdacordao})")
            page.wait_for_selector(f"#{div_id}", state="visible", timeout=10_000)
            texto = page.inner_text(f"#{div_id}").strip()

            results.append({
                "numero_processo": numero_processo,
                "cdacordao":       cdacordao,
                "classe":          classe,
                "texto":           texto,
            })
            seen.add(numero_processo)
            restantes -= 1
            print(f"    ✓ {numero_processo} | {classe} ({len(texto)} chars)")

            # Close popup if a close button is present
            try:
                close_btn = page.query_selector(
                    "a.fechar, button.fechar, "
                    "a[onclick*='fechar'], a[onclick*='close'], "
                    "a[title='Fechar'], button[title='Fechar'], "
                    ".ui-dialog-titlebar-close"
                )
                if close_btn:
                    close_btn.click()
                    time.sleep(0.3)
            except Exception:
                pass

        except PWTimeout:
            print(f"    ✗ Timeout on {numero_processo} — div #{div_id} did not appear")
        except Exception as exc:
            print(f"    ✗ Error on {numero_processo}: {exc}")

        time.sleep(DELAY_S)

    return results


def go_to_next_page(page) -> bool:
    """Clicks the 'Next' pagination link if available. Returns True on success."""
    try:
        next_link = page.query_selector(
            "a:has-text('Próxima'), a:has-text('>>'), a[title='Próxima página']"
        )
        if next_link:
            next_link.click()
            page.wait_for_load_state("networkidle", timeout=20_000)
            time.sleep(DELAY_S)
            return True
    except Exception:
        pass
    return False


def run_search(page, term: str, max_pages: int, seen: set) -> list[dict]:
    """
    Navigates to the search page, submits the given term, and scrapes
    up to max_pages pages or LIMIT_PER_SEARCH ementas, whichever comes first.

    Args:
        page:      Playwright page object.
        term:      Search term to enter in the free-text field.
        max_pages: Maximum number of result pages to iterate.
        seen:      Global deduplication set (mutated in place).

    Returns:
        List of collected ementa dicts for this search term.
    """
    print(f"\n{'═' * 60}")
    print(f"  SEARCH: {term}")
    print(f"{'═' * 60}")

    collected: list[dict] = []

    # Always start from the base URL to reset any previous search state
    page.goto(BASE_URL, wait_until="networkidle", timeout=30_000)
    time.sleep(1)

    # Fill in the free-text search field
    field = page.wait_for_selector(
        "input[name='dados.buscaInteiroTeor']", timeout=15_000
    )
    field.click(click_count=3)
    field.type(term)

    # Submit — try the search button first, fall back to Enter
    try:
        btn = page.query_selector(
            "input[type='submit'][value*='Pesquisar'], "
            "button:has-text('Pesquisar')"
        )
        if btn:
            btn.click()
        else:
            field.press("Enter")
    except Exception:
        field.press("Enter")

    page.wait_for_load_state("networkidle", timeout=30_000)
    time.sleep(1)
    print("  Results loaded.")

    for page_num in range(1, max_pages + 1):
        remaining = LIMIT_PER_SEARCH - len(collected)
        if remaining <= 0:
            print(f"  Limit of {LIMIT_PER_SEARCH} reached for this search term.")
            break

        print(f"\n  ── Page {page_num} ──")
        ementas = collect_ementas_from_page(page, remaining, seen)
        collected.extend(ementas)

        if page_num < max_pages:
            if not go_to_next_page(page):
                print("  No more pages available.")
                break

    print(f"\n  Collected for '{term}': {len(collected)}")
    return collected


def main(paginas: int = 1, headless: bool = True):
    all_ementas: list[dict] = []
    seen: set[str] = set()  # global deduplication across all search terms

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page    = context.new_page()

        for term in SEARCH_TERMS:
            ementas = run_search(page, term, paginas, seen)
            all_ementas.extend(ementas)

        browser.close()

    # Save results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_ementas, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'═' * 60}")
    print(f"✅ {len(all_ementas)} ementa(s) saved to '{OUTPUT_FILE}'")
    for classe, count in Counter(e["classe"] for e in all_ementas).items():
        print(f"   • {classe or '(unknown)'}: {count}")
    print(f"{'═' * 60}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TJMS eSAJ ementa scraper")
    parser.add_argument(
        "--paginas", type=int, default=1,
        help="Number of result pages to scrape per search term (default: 1)"
    )
    parser.add_argument(
        "--headless", type=lambda x: x.lower() != "false", default=True,
        help="Run headless browser: true/false (default: true)"
    )
    args = parser.parse_args()
    main(paginas=args.paginas, headless=args.headless)