"""report.html → PDF（Chromium/Playwright で印刷）。  python report/make_pdf.py [出力ファイル名]"""
import os, sys
from playwright.sync_api import sync_playwright

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "説明資料_PAI最終課題_ACCOUNT.pdf")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto("file://" + os.path.join(here, "report.html"))
    pg.wait_for_timeout(500)
    pg.pdf(path=out, format="A4", print_background=True,
           margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
           display_header_footer=True, header_template="<div></div>",
           footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#666">'
                           '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    b.close()
print("saved", out)
