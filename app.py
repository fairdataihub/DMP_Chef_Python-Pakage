# ===============================================================
# app.py — NIH DMP Generator (FastAPI) + Working Download Buttons
# - Uses pipeline output folders directly (no hard-coded paths)
# - URL-encodes filenames so spaces don't break download links
# ===============================================================

import re
from pathlib import Path
from html import escape
from urllib.parse import quote

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from src.core_pipeline_UI import DMPPipeline

app = FastAPI()

pipeline = DMPPipeline(config_path="config/config.yaml", force_rebuild_index=False)


def safe_filename(title: str) -> str:
    """Match the same safe-title rule used in core_pipeline_UI.py."""
    return re.sub(r'[\\/*?:"<>|]', "_", (title or "").strip()).strip()


@app.get("/", response_class=HTMLResponse)
async def form_page():
    return render_form()


@app.post("/", response_class=HTMLResponse)
async def generate_dmp(
    request: Request,
    title: str = Form(...),
    research_context: str = Form(""),
    data_types: str = Form(""),
    data_source: str = Form(""),
    human_subjects: str = Form(""),
    consent_status: str = Form(""),
    data_volume: str = Form(""),
):
    form_inputs = {
        "research_context": research_context,
        "data_types": data_types,
        "data_source": data_source,
        "human_subjects": human_subjects,
        "consent_status": consent_status,
        "data_volume": data_volume,
    }

    md_text = pipeline.generate_dmp(title, form_inputs)
    return render_form(result=md_text, title=title)


# -------------------- Download endpoints -----------------------

@app.get("/download/json/{title}")
async def download_json(title: str):
    st = safe_filename(title)
    json_dir = getattr(pipeline, "output_json", Path("data/outputs/json"))
    json_path = Path(json_dir) / f"{st}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"JSON not found: {json_path}")
    return FileResponse(str(json_path), media_type="application/json", filename=f"{st}.json")


@app.get("/download/docx/{title}")
async def download_docx(title: str):
    st = safe_filename(title)
    docx_dir = getattr(pipeline, "output_docx", Path("data/outputs/docx"))
    docx_path = Path(docx_dir) / f"{st}.docx"
    if not docx_path.exists():
        raise HTTPException(status_code=404, detail=f"DOCX not found: {docx_path}")
    return FileResponse(
        str(docx_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{st}.docx",
    )


@app.get("/download/md/{title}")
async def download_md(title: str):
    st = safe_filename(title)
    md_dir = getattr(pipeline, "output_md", Path("data/outputs/markdown"))
    md_path = Path(md_dir) / f"{st}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Markdown not found: {md_path}")
    return FileResponse(str(md_path), media_type="text/markdown", filename=f"{st}.md")


# -------------------- HTML rendering ---------------------------

def render_form(result: str = "", title: str = "") -> str:
    st = safe_filename(title)
    t = escape(title or "")
    st_url = quote(st)  # important: spaces -> %20

    result_html = ""
    if result:
        result_html = f"""
        <hr style="margin: 40px 0;">
        <h2>✅ NIH DMP Generated for: <i>{t}</i></h2>

        <div style="display:flex; gap:10px; margin: 12px 0 18px 0; flex-wrap: wrap;">
            <a href="/download/json/{st_url}"
               style="display:inline-block; padding:10px 14px; background:#1f7a1f; color:#fff; text-decoration:none; border-radius:6px;">
               ⬇️ Download JSON
            </a>

            <a href="/download/docx/{st_url}"
               style="display:inline-block; padding:10px 14px; background:#6a1b9a; color:#fff; text-decoration:none; border-radius:6px;">
               ⬇️ Download DOCX
            </a>

            <a href="/download/md/{st_url}"
               style="display:inline-block; padding:10px 14px; background:#444; color:#fff; text-decoration:none; border-radius:6px;">
               ⬇️ Download Markdown
            </a>
        </div>

        <pre style="white-space: pre-wrap; background:#f8f8f8; padding:15px; border-radius:8px; border:1px solid #eee;">
{escape(result)}
        </pre>
        """

    return f"""
    <html>
    <head>
        <title>NIH DMP Generator</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; max-width: 900px; }}
            textarea, input {{ width: 100%; padding: 8px; margin-top: 4px; margin-bottom: 16px; }}
            textarea {{ height: 100px; }}
            button {{ padding: 10px 20px; background-color: #0078d7; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            button:hover {{ background-color: #005fa3; }}
            pre {{ font-family: Consolas, monospace; font-size: 14px; line-height: 1.5; }}
        </style>
    </head>
    <body>
        <h1>🧠 NIH Data Management Plan Generator</h1>

        <form action="/" method="post">
            <label><b>Project Title:</b></label>
            <input type="text" name="title" required value="{t}"><br>

            <label><b>Brief summary of the research context:</b></label>
            <textarea name="research_context" placeholder="Describe the scientific goals and objectives..."></textarea><br>

            <label><b>Types of data to be collected:</b></label>
            <textarea name="data_types" placeholder="Genomic data, survey responses, imaging, etc."></textarea><br>

            <label><b>Source of data:</b></label>
            <textarea name="data_source" placeholder="Human participants, sensors, clinical instruments..."></textarea><br>

            <label><b>Human subjects involvement:</b></label>
            <textarea name="human_subjects" placeholder="Does the study involve human participants?"></textarea><br>

            <label><b>Data sharing consent status (if applicable):</b></label>
            <textarea name="consent_status" placeholder="Broad sharing approved? Any restrictions?"></textarea><br>

            <label><b>Estimated data volume, modality, and format:</b></label>
            <textarea name="data_volume" placeholder="Structured/unstructured data, formats (CSV, DICOM, etc.)"></textarea><br>

            <button type="submit">Generate DMP</button>
        </form>

        {result_html}
    </body>
    </html>
    """
