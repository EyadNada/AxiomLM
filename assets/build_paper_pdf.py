"""
Renders docs/assets/AxiomLM_Research_Paper_Draft_v2.md to a publication-ready HTML and PDF
manuscript (docs/assets/AxiomLM_Research_Paper_Draft_v2.pdf and docs/assets/AxiomLM_Research_Paper_Draft.pdf).
"""

import os
import re
import shutil
import subprocess
import markdown

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(PROJECT_ROOT, "docs/assets/AxiomLM_Research_Paper_Draft_v2.md")
HTML_PATH = os.path.join(
    PROJECT_ROOT, "docs/assets/AxiomLM_Research_Paper_Draft_v2.html"
)
PDF_V2_PATH = os.path.join(
    PROJECT_ROOT, "docs/assets/AxiomLM_Research_Paper_Draft_v2.pdf"
)
PDF_MAIN_PATH = os.path.join(
    PROJECT_ROOT, "docs/assets/AxiomLM_Research_Paper_Draft.pdf"
)
PDF_DRAFT1_BACKUP = os.path.join(
    PROJECT_ROOT, "docs/assets/AxiomLM_Research_Paper_Draft_1_Backup.pdf"
)


def build_pdf():
    if os.path.exists(PDF_MAIN_PATH) and not os.path.exists(PDF_DRAFT1_BACKUP):
        shutil.copy(PDF_MAIN_PATH, PDF_DRAFT1_BACKUP)
        print(f"Backed up Draft 1 PDF to: {PDF_DRAFT1_BACKUP}")

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert relative ../../assets/ image paths to absolute file:// paths so Chrome renders them cleanly
    assets_abs = "file://" + os.path.join(PROJECT_ROOT, "assets") + "/"
    md_text = md_text.replace("../../assets/", assets_abs)

    # Protect display and inline math before markdown conversion
    math_blocks = []

    def save_math(match):
        math_blocks.append(match.group(0))
        return f"@@MATH_BLOCK_{len(math_blocks) - 1}@@"

    md_text_protected = re.sub(r"\$\$.*?\$\$", save_math, md_text, flags=re.DOTALL)
    md_text_protected = re.sub(r"\$[^\$\n]+\$", save_math, md_text_protected)

    html_body = markdown.markdown(
        md_text_protected,
        extensions=["tables", "fenced_code"],
    )

    for idx, block in enumerate(math_blocks):
        html_body = html_body.replace(f"@@MATH_BLOCK_{idx}@@", block)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The Axiom Engine: Research Paper (Draft 2)</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  @page {{
    size: letter;
    margin: 20mm 18mm 20mm 18mm;
  }}
  body {{
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.52;
    color: #111827;
    max-width: 820px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 18pt;
    line-height: 1.25;
    text-align: center;
    margin-bottom: 6px;
    color: #0f172a;
  }}
  h2 {{
    font-size: 12.5pt;
    margin-top: 22px;
    margin-bottom: 8px;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 4px;
    color: #1e293b;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 11pt;
    margin-top: 14px;
    margin-bottom: 6px;
    color: #1e293b;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 9px;
    text-align: justify;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 5px 7px;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    font-weight: 700;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  img {{
    display: block;
    max-width: 92%;
    margin: 12px auto 4px auto;
    page-break-inside: avoid;
  }}
  code {{
    font-family: "SFMono-Regular", Consolas, Menlo, monospace;
    font-size: 8.8pt;
    background-color: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
  }}
  hr {{
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 16px 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Wrote HTML manuscript: {HTML_PATH}")

    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome_bin):
        cmd = [
            chrome_bin,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=8000",
            f"--print-to-pdf={PDF_V2_PATH}",
            f"file://{HTML_PATH}",
        ]
        subprocess.run(cmd, check=True)
        shutil.copy(PDF_V2_PATH, PDF_MAIN_PATH)
        print(f"Generated PDF manuscripts:\n  • {PDF_V2_PATH}\n  • {PDF_MAIN_PATH}")


if __name__ == "__main__":
    build_pdf()
