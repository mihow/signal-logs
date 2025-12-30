"""
Simple web UI for viewing passive monitoring summaries.

Run with: uv run python summary_viewer.py
Access at: http://localhost:8080
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


app = FastAPI(title="Auto Dispatch - Summary Viewer")

SUMMARY_DIR = Path(os.getenv("SUMMARY_OUTPUT_DIR", "summaries/"))


def parse_summary_file(filepath: Path) -> dict[str, Any]:
    """Parse a summary file and extract metadata."""
    with open(filepath) as f:
        content = f.read()

    # Extract metadata from header section
    lines = content.split("\n")
    metadata = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "content": content,
        "session_start": None,
        "session_end": None,
        "duration": None,
        "speakers": None,
        "callsigns": [],
        "topics": [],
        "summary": None,
        "digital_signals": [],
        "transcript_count": 0,
    }

    # Parse header
    for line in lines:
        if line.startswith("Session Start:"):
            metadata["session_start"] = line.split(":", 1)[1].strip()
        elif line.startswith("Session End:"):
            metadata["session_end"] = line.split(":", 1)[1].strip()
        elif line.startswith("Duration:"):
            metadata["duration"] = line.split(":", 1)[1].strip()
        elif line.startswith("Speakers:"):
            metadata["speakers"] = line.split(":", 1)[1].strip()
        elif line.startswith("Call Signs:"):
            callsigns_str = line.split(":", 1)[1].strip()
            if callsigns_str != "None detected":
                metadata["callsigns"] = [c.strip() for c in callsigns_str.split(",")]
        elif line.startswith("- ") and "Topics:" in content[:content.find(line)]:
            # Topic line
            topic = line[2:].strip()
            if topic and topic not in metadata["topics"]:
                metadata["topics"].append(topic)

    # Extract summary section
    if "Summary" in content and "=========" in content:
        summary_start = content.find("Summary\n=================================================\n")
        if summary_start != -1:
            summary_start = summary_start + len("Summary\n=================================================\n")
            summary_end = content.find("\n=================================================", summary_start)
            if summary_end != -1:
                metadata["summary"] = content[summary_start:summary_end].strip()

    # Extract digital signals if present
    if "Digital Signals Detected" in content:
        signals_start = content.find("Digital Signals Detected (Not Decoded)\n=================================================\n")
        if signals_start != -1:
            signals_start = signals_start + len("Digital Signals Detected (Not Decoded)\n=================================================\n")
            signals_end = content.find("\n=================================================", signals_start)
            if signals_end != -1:
                signals_section = content[signals_start:signals_end].strip()
                for line in signals_section.split("\n"):
                    if line.strip():
                        metadata["digital_signals"].append(line.strip())

    # Count transcript entries
    metadata["transcript_count"] = content.count('Speaker ')

    # Parse timestamp from filename for sorting
    if filepath.name.startswith("202"):
        timestamp_str = filepath.name.split("_")[0]
        try:
            metadata["file_timestamp"] = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
        except ValueError:
            metadata["file_timestamp"] = datetime.fromtimestamp(filepath.stat().st_mtime)
    else:
        metadata["file_timestamp"] = datetime.fromtimestamp(filepath.stat().st_mtime)

    return metadata


@app.get("/", response_class=HTMLResponse)
async def index():
    """Main page - list all summaries."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Auto Dispatch - Summary Viewer</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #2a2a3e;
        }
        h1 {
            color: #00d4ff;
            font-size: 2em;
            margin-bottom: 10px;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        .subtitle { color: #888; font-size: 0.9em; }
        .summary-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .summary-card:hover {
            background: #222;
            border-color: #00d4ff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        .timestamp {
            color: #00d4ff;
            font-size: 1.1em;
            font-weight: bold;
        }
        .duration {
            color: #888;
            font-size: 0.85em;
            background: #2a2a2a;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .callsigns {
            color: #ff9500;
            font-weight: bold;
            margin: 8px 0;
            font-size: 0.95em;
        }
        .topics {
            color: #64d2ff;
            margin: 8px 0;
            font-size: 0.9em;
        }
        .summary-preview {
            color: #ccc;
            margin-top: 12px;
            font-size: 0.9em;
            line-height: 1.5;
        }
        .meta {
            display: flex;
            gap: 15px;
            margin-top: 12px;
            font-size: 0.85em;
            color: #888;
        }
        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .badge {
            background: #2a2a3e;
            color: #00d4ff;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }
        .digital-signal {
            background: #3a1a3e;
            border-left: 3px solid #ff00ff;
            padding: 4px 8px;
            margin-top: 8px;
            font-size: 0.85em;
            border-radius: 4px;
        }
        .no-summaries {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        a { color: #00d4ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📻 Auto Dispatch - Summary Viewer</h1>
            <div class="subtitle">Passive Radio Monitoring Summaries</div>
        </header>

        <div id="summaries" class="loading">Loading summaries...</div>
    </div>

    <script>
        async function loadSummaries() {
            try {
                const response = await fetch('/api/summaries');
                const summaries = await response.json();

                const container = document.getElementById('summaries');

                if (summaries.length === 0) {
                    container.innerHTML = '<div class="no-summaries">No summaries found. Run the passive bot to generate summaries.</div>';
                    return;
                }

                container.innerHTML = '';

                summaries.forEach(s => {
                    const card = document.createElement('div');
                    card.className = 'summary-card';

                    const callsignsHtml = s.callsigns.length > 0
                        ? `<div class="callsigns">📡 ${s.callsigns.join(', ')}</div>`
                        : '';

                    const topicsHtml = s.topics.length > 0
                        ? `<div class="topics">🏷️ ${s.topics.join(' • ')}</div>`
                        : '';

                    const digitalSignalsHtml = s.digital_signals.length > 0
                        ? s.digital_signals.map(sig => `<div class="digital-signal">📊 ${sig}</div>`).join('')
                        : '';

                    card.innerHTML = `
                        <div class="card-header">
                            <div class="timestamp">${new Date(s.file_timestamp).toLocaleString()}</div>
                            <div class="duration">${s.duration || 'Unknown'}</div>
                        </div>
                        ${callsignsHtml}
                        ${topicsHtml}
                        <div class="summary-preview">${s.summary || 'No summary available'}</div>
                        ${digitalSignalsHtml}
                        <div class="meta">
                            <div class="meta-item">
                                <span>👥</span>
                                <span>${s.speakers || '0'}</span>
                            </div>
                            <div class="meta-item">
                                <span>💬</span>
                                <span>${s.transcript_count} messages</span>
                            </div>
                            ${s.digital_signals.length > 0 ? `
                                <div class="meta-item">
                                    <span class="badge">${s.digital_signals.length} digital signal(s)</span>
                                </div>
                            ` : ''}
                        </div>
                    `;

                    card.onclick = () => window.location = '/summary/' + s.filename;
                    container.appendChild(card);
                });
            } catch (error) {
                document.getElementById('summaries').innerHTML =
                    '<div class="no-summaries">Error loading summaries: ' + error.message + '</div>';
            }
        }

        // Load summaries on page load
        loadSummaries();

        // Auto-refresh every 30 seconds
        setInterval(loadSummaries, 30000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.get("/api/summaries")
async def list_summaries():
    """API endpoint: List all summary files with metadata."""
    if not SUMMARY_DIR.exists():
        return []

    summaries = []
    for file in sorted(SUMMARY_DIR.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            metadata = parse_summary_file(file)
            summaries.append(metadata)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            continue

    return summaries


@app.get("/summary/{filename}")
async def view_summary(filename: str):
    """View full summary file."""
    filepath = SUMMARY_DIR / filename

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Summary not found")

    metadata = parse_summary_file(filepath)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{filename} - Auto Dispatch</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .back-link {{
            color: #00d4ff;
            text-decoration: none;
            margin-bottom: 20px;
            display: inline-block;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        pre {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← Back to summaries</a>
        <pre>{metadata['content']}</pre>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    print(f"📻 Starting Summary Viewer on http://localhost:8080")
    print(f"📁 Serving summaries from: {SUMMARY_DIR.absolute()}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
