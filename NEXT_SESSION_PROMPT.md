# Next Session: Passive Bot Summary Testing & Web UI

## Context from Previous Session (2025-12-30)

### ✅ Completed Work
**Session Focus:** Fixed critical StartFrame bug in passive listening bot and verified production functionality.

**Major Accomplishments:**
1. **Fixed StartFrame Error** - `passive_processors.py:159`
   - Added `await super().process_frame(frame, direction)` call
   - Zero errors in production testing

2. **Fixed Metadata Passing** - `passive_processors.py:27,113-120,154,157-182`
   - Changed from TextFrame metadata to direct reference passing
   - `TranscriptBatchProcessor` now holds reference to `SummaryWriter`

3. **Production Testing Verified**
   - 139 transcriptions captured across 3 summary files
   - 13 speakers detected via pause-based algorithm
   - All files created successfully in `summaries/` directory
   - Zero pipeline errors during 15-minute live test

**Key Files Modified:**
- `passive_processors.py` - Core processors (batch, summary writer)
- `bot_passive.py` - Main passive bot server
- `PLAN-passive-listening-bot.md` - Updated with fixes and roadmap

**Current Bot Status:**
- ✅ Server runs on http://localhost:7860
- ✅ Transcription working (Whisper STT)
- ✅ Summary files being created
- ⚠️ Summaries may need refinement (casual conversation vs radio content)

---

## 🎯 This Session's Objectives

### Primary Goal
**Test and validate summary generation quality, then build a simple web UI to view summaries.**

### Task Breakdown

#### Task 1: Evaluate Current Summaries (30 min)

**What to Check:**
1. **Review Existing Summary Files**
   - Location: `summaries/` directory
   - Files created:
     - `20251230-101546_conversation.txt` (63 transcriptions)
     - `20251230-102049_conversation.txt` (60 transcriptions)
     - `20251230-102607_conversation.txt` (16 transcriptions)

2. **Analyze Summary Quality**
   - Are topics being extracted correctly?
   - Is call sign detection working? (Format: W1ABC, K2XYZ, etc.)
   - Are speaker counts accurate?
   - Is the LLM summary field populated? (Currently shows "No summary available")

3. **Test with Radio Content**
   - Available recordings: `recordings/` directory
     - `merged_20251229_010801.wav` (575.8s, 22050Hz)
     - `merged_20251229_014527.wav`
     - `merged_20251229_233629.wav`
     - `merged_20251230_000117.wav`
     - `merged_20251230_001847.wav`
   - Test file: `test2.wav` (11.5s, 16000Hz) - Good for quick tests

4. **Identify Issues**
   - Why is LLM not generating summaries? (Logging shows: "LLM response received but no batch metadata available")
   - Is the batch processor actually triggering?
   - Are transcripts being batched correctly?

**Commands to Run:**
```bash
# Check current summaries
ls -lh summaries/
cat summaries/*.txt | head -100

# Check bot logs for batch processing
grep -i "batch\|summary" /tmp/claude/.../b060bbc.output | tail -50

# Test with a recording (if bot not running)
# First: Stop current bot if running
pkill -f bot_passive.py

# Option A: Use the integration test
uv run python test_passive_integration.py

# Option B: Use the audio file test (needs updates)
uv run python test_passive_bot.py test2.wav
```

#### Task 2: Fix Summary Generation (if broken) (45 min)

**Potential Issues Identified:**
1. **Batch processor not wired correctly**
   - Check: `bot_passive.py:91-95` - Is `batch_processor` connected to transcript events?
   - Currently: Batch processor created but not receiving transcripts!
   - Fix: Need to connect transcript processor events to batch processor

2. **LLM receiving transcripts but not batch metadata**
   - Warning seen: "LLM response received but no batch metadata available"
   - This means LLM is generating responses, but `SummaryWriter` doesn't have metadata
   - Likely cause: Batch processor never calls `_process_batch()`

**Action Items:**
- [ ] Review `bot_passive.py` - How are transcripts getting to batch processor?
- [ ] Add event handler for transcript updates → batch processor
- [ ] Verify `batch_processor._process_batch()` is being called
- [ ] Test batch trigger with manual transcript injection

**Expected Flow:**
```
Audio → Whisper → TranscriptProcessor → (EVENT) → TranscriptBatchProcessor
                                                          ↓
                                                    _process_batch()
                                                          ↓
                                    Set metadata in SummaryWriter
                                                          ↓
                                    Queue LLMMessagesFrame to task
                                                          ↓
                                    LLM generates response → SummaryWriter
                                                          ↓
                                                   Write summary file
```

#### Task 3: Build Simple Web UI (1-2 hours)

**Reference Implementation:**
- Framework location: `../pipecat-test-web/` directory
- Check what's already there for structure/patterns

**Requirements:**
1. **Summary List Page** (`/summaries`)
   - Display all summary files from `summaries/` directory
   - Show: timestamp, duration, speaker count, topics, call signs
   - Sort by date (newest first)
   - Click to view details

2. **Summary Detail Page** (`/summary/<filename>`)
   - Full transcript with timestamps
   - Speaker breakdown
   - Topics and call signs highlighted
   - Metadata summary

3. **Live Monitor Page** (`/monitor`) (Optional - if time)
   - Current batch transcripts (live)
   - Time until next summary
   - WebSocket for real-time updates

**Implementation Approach:**
```python
# Add to bot_passive.py or create new summary_api.py

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
import json

app = FastAPI()

@app.get("/api/summaries")
async def list_summaries():
    """Return list of all summary files with metadata"""
    summaries = []
    for file in Path("summaries").glob("*.txt"):
        # Parse summary file for metadata
        with open(file) as f:
            content = f.read()
            # Extract metadata from file
            summaries.append({
                "filename": file.name,
                "path": str(file),
                "timestamp": file.stat().st_mtime,
                # ... parse content for topics, speakers, etc.
            })
    return sorted(summaries, key=lambda x: x["timestamp"], reverse=True)

@app.get("/api/summary/{filename}")
async def get_summary(filename: str):
    """Return full summary content"""
    file_path = Path("summaries") / filename
    if not file_path.exists():
        return {"error": "Not found"}

    with open(file_path) as f:
        content = f.read()

    # Parse and return structured data
    return {"filename": filename, "content": content}

@app.get("/")
async def index():
    """Serve simple HTML UI"""
    return HTMLResponse(open("summary_viewer.html").read())
```

**HTML Template Structure:**
```html
<!-- summary_viewer.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Passive Bot Summaries</title>
    <style>
        body { font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .summary-card { border: 1px solid #ccc; margin: 10px 0; padding: 15px; }
        .summary-card:hover { background: #f5f5f5; cursor: pointer; }
        .timestamp { color: #666; font-size: 0.9em; }
        .topics { color: #0066cc; }
        .callsigns { color: #cc6600; font-weight: bold; }
        .transcript { white-space: pre-wrap; background: #f9f9f9; padding: 10px; }
    </style>
</head>
<body>
    <h1>📻 Passive Bot Summaries</h1>
    <div id="summaries"></div>

    <script>
        // Fetch and display summaries
        fetch('/api/summaries')
            .then(r => r.json())
            .then(summaries => {
                const container = document.getElementById('summaries');
                summaries.forEach(s => {
                    const card = document.createElement('div');
                    card.className = 'summary-card';
                    card.innerHTML = `
                        <div class="timestamp">${new Date(s.timestamp * 1000).toLocaleString()}</div>
                        <div class="topics">Topics: ${s.topics || 'None'}</div>
                        <div class="callsigns">Call Signs: ${s.callsigns || 'None'}</div>
                        <div>Speakers: ${s.speaker_count || 0} | Duration: ${s.duration || 'Unknown'}</div>
                    `;
                    card.onclick = () => window.location = '/summary/' + s.filename;
                    container.appendChild(card);
                });
            });
    </script>
</body>
</html>
```

---

## 📁 Key Files & Locations

**Core Code:**
- `bot_passive.py:1-150` - Main bot server
- `passive_processors.py:1-291` - TranscriptBatchProcessor, SummaryWriter
- `PLAN-passive-listening-bot.md` - Complete documentation and roadmap

**Test Files:**
- `test_passive_integration.py` - Full pipeline test (WORKING)
- `test_passive_bot.py` - Audio file test (needs fixing)
- `verify_passive_bot.py` - Production verification script

**Data Directories:**
- `summaries/` - Generated summary files (3 files currently)
- `recordings/` - Test audio recordings (5 WAV files)
- `test_summaries/` - Integration test output

**Reference Code:**
- `../pipecat-test-web/` - Web UI framework reference

**Environment:**
- Bot runs on: http://localhost:7860
- Client UI: http://localhost:7860/client
- Ollama: http://localhost:11434 (llama3.3 model)
- Whisper: CPU mode, base model

---

## 🔍 Investigation Questions

**Answer these during the session:**

1. **Is the batch processor receiving transcripts?**
   - Add logging to `TranscriptBatchProcessor.add_transcript()`
   - Check if `_current_batch` is being populated

2. **Is `_process_batch()` being called?**
   - Add logging at start of `_process_batch()`
   - Check batch window timer logic

3. **Why does SummaryWriter not have metadata?**
   - Verify `batch_processor` has reference to `summary_writer`
   - Check if `_pending_batch_metadata` is being set
   - Add logging when metadata is set

4. **Is the LLM generating proper responses?**
   - Check LLM prompt in logs
   - Verify response format matches expected: "TOPICS: ... CALLSIGNS: ... SUMMARY: ..."

5. **What's in the existing summary files?**
   - Read through `summaries/*.txt`
   - Check if "No summary available" is in all files
   - Look for any successfully parsed summaries

---

## 🚀 Success Criteria

**By end of session, should have:**

✅ **Summary Generation Validated**
- [ ] Identified why summaries aren't working (if broken)
- [ ] Fixed batch processor wiring
- [ ] Generated at least 1 complete summary with topics/callsigns
- [ ] Tested with both live mic and recording file

✅ **Basic Web UI Working**
- [ ] List page shows all summaries
- [ ] Can click to view full transcript
- [ ] Metadata is displayed correctly
- [ ] UI is accessible on localhost

✅ **Documentation Updated**
- [ ] Added findings to PLAN-passive-listening-bot.md
- [ ] Noted any new bugs discovered
- [ ] Updated "Next Steps" section

---

## 💡 Tips & Reminders

**Debugging Approach:**
1. Start by reading existing summary files
2. Run integration test to see current behavior
3. Add strategic logging before making changes
4. Test incrementally (don't change everything at once)

**Web UI Development:**
1. Start simple - just list files first
2. Use browser to test (no need for fancy framework)
3. Can integrate with existing bot server or run separately
4. Reference pipecat-test-web for patterns

**Common Pitfalls:**
- Don't forget to restart bot after code changes
- Remember to check logs for errors
- Summary files might be cached - check timestamps
- Whisper needs 16kHz audio - resample if needed

**Quick Test Commands:**
```bash
# Kill existing bot
pkill -f bot_passive.py

# Start fresh bot with logging
uv run bot_passive.py 2>&1 | tee bot.log

# Monitor summaries being created
watch -n 1 'ls -lh summaries/ && tail -20 summaries/*.txt'

# Test integration (includes batch trigger)
uv run python test_passive_integration.py
```

---

## 📚 References

**Pipecat Concepts:**
- FrameProcessor lifecycle: StartFrame → process → EndFrame
- Event handlers: `@processor.event_handler("event_name")`
- Pipeline flow: frames flow downstream via `push_frame()`

**Current Bot Architecture:**
```
WebRTC/Audio → Whisper STT → TranscriptProcessor
                                      ↓
                              (transcripts to LLM context)
                                      ↓
                              LLMContextAggregator
                                      ↓
                              OLLamaLLM
                                      ↓
                              SummaryWriter
```

**Missing Connection:**
```
TranscriptProcessor → (NEED EVENT HANDLER) → TranscriptBatchProcessor
                                                      ↓
                                              _process_batch()
                                                      ↓
                                          Set metadata in SummaryWriter
```

Good luck! 🚀
