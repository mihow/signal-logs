# Next Session: Digital Signal Detection & Web UI Enhancements

## Context from Previous Session (2025-12-30)

### ✅ Completed Work

**Session Focus:** Fixed LLM summary generation and built web UI for viewing logs

**Major Accomplishments:**

1. **Replaced Brittle Regex Parsing with Structured JSON** - `passive_processors.py:27-39,233-282`
   - Added Pydantic models: `RadioSummary`, `DigitalSignal`
   - Enabled Ollama JSON mode: `bot_passive.py:57-59`
   - LLM now returns valid JSON with topics, callsigns, summary, digital_signals
   - Strips markdown code fences automatically (```json...```)
   - Wait for `LLMFullResponseEndFrame` instead of checking string markers
   - Graceful fallback on parse failures

2. **Built Web UI for Viewing Summaries** - `summary_viewer.py:1-403`
   - FastAPI server on port 8080
   - Dark themed monospace UI optimized for radio monitoring
   - Lists all summaries with metadata (callsigns, topics, duration, digital signals)
   - Click cards to view full transcript
   - Auto-refreshes every 30 seconds
   - Parses summary files and extracts structured metadata

3. **Documented Digital Signal Detection** - `docs/claude/planning/DIGITAL-SIGNAL-DETECTION.md`
   - Three audio routing approaches for Pipecat
   - Example `SignalDetector` implementation (amplitude + FFT)
   - TorchSig ML-based classification approach
   - Integration guide with batch processor
   - Testing resources and implementation checklist

**Key Files Modified:**
- `passive_processors.py` - JSON parsing, Pydantic models, digital signal support
- `bot_passive.py` - Enabled JSON mode for Ollama
- `summary_viewer.py` - NEW: Web UI for viewing summaries
- `docs/claude/planning/DIGITAL-SIGNAL-DETECTION.md` - NEW: Future work planning

**Current Status:**
- ✅ All integration tests passing
- ✅ Summary generation working with proper topics, callsigns, and summaries
- ✅ Web UI running at http://localhost:8080
- ✅ Digital signal logging supported (waiting for detection implementation)
- ⚠️ Digital signal detection NOT yet implemented (logged but not detected)

---

## 🎯 This Session's Objectives

### Primary Goal
**Implement basic digital signal detection and test with real radio recordings.**

### Task Breakdown

#### Task 1: Review and Test Current System (15 min)

**What to Check:**
1. **Run the passive bot and web UI**
   ```bash
   # Terminal 1: Start passive bot
   uv run bot_passive.py

   # Terminal 2: Start web UI
   uv run python summary_viewer.py

   # Terminal 3: Open browser
   xdg-open http://localhost:8080
   ```

2. **Test with existing recordings**
   - Recordings available in `recordings/` directory
   - Test files: `merged_20251229_*.wav`
   - Check if summaries are being generated correctly

3. **Review web UI functionality**
   - Can you see all summaries?
   - Do digital signals display (if any)?
   - Does the auto-refresh work?
   - Can you click into full transcripts?

#### Task 2: Implement Basic Signal Detector (1-2 hours)

**Reference:** `docs/claude/planning/DIGITAL-SIGNAL-DETECTION.md`

**Implementation Steps:**

1. **Create SignalDetector Processor** - New file: `signal_detector.py`
   ```python
   class SignalDetector(FrameProcessor):
       """Detect digital signals based on audio characteristics."""

       # Use amplitude + spectral analysis
       # Emit DigitalSignalFrame when detected
   ```

2. **Create DigitalSignalFrame** - Add to `passive_processors.py` or new `custom_frames.py`
   ```python
   class DigitalSignalFrame(DataFrame):
       timestamp: str
       duration_seconds: float
       signal_type: str | None
   ```

3. **Wire Up in Pipeline** - `bot_passive.py:103-111`
   ```python
   pipeline = Pipeline([
       transport.input(),
       stt,
       signal_detector,  # ADD THIS
       transcript_processor.user(),
       ...
   ])
   ```

4. **Add Event Handler** - `bot_passive.py` after line 135
   ```python
   @signal_detector.event_handler("on_digital_signal_detected")
   async def on_digital_signal(detector, frame):
       await batch_processor.add_digital_signal(frame)
   ```

5. **Update TranscriptBatchProcessor** - `passive_processors.py:42`
   ```python
   def __init__(self, ...):
       self._digital_signals = []  # Add this

   async def add_digital_signal(self, frame):
       """Collect digital signal detections."""
       self._digital_signals.append({...})

   async def _process_batch(self):
       # Include in metadata
       self._summary_writer._pending_batch_metadata["digital_signals"] = ...
   ```

**Testing:**
- Use `test2.wav` (11.5s) for quick tests
- Use `recordings/merged_*.wav` for real radio content
- Check web UI to see if digital signals appear

#### Task 3: Tune Detection Parameters (30-60 min)

**Goal:** Minimize false positives while catching real digital signals

**Approach:**
1. Test with known digital signal recordings (FT8, RTTY, PSK31)
   - Download samples from https://www.sigidwiki.com/wiki/
   - Test detection threshold and duration settings

2. Test with voice-only recordings
   - Should NOT detect digital signals in normal conversation
   - Adjust spectral flatness and harmonic ratio thresholds

3. Test with mixed content (voice + digital)
   - Should detect only the digital portions
   - Verify timestamps and durations are accurate

**Parameters to Tune:**
- `threshold_db`: Minimum amplitude to consider (default: -20 dB)
- `min_duration_sec`: Minimum signal length (default: 0.5s)
- `spectral_flatness`: Threshold for "noise-like" signals (default: 0.5)
- `harmonic_ratio`: Threshold for non-harmonic content (default: 0.3)

#### Task 4: Web UI Enhancements (Optional - if time)

**Possible Improvements:**
1. **Filtering and Search**
   - Filter by callsign
   - Filter by topic
   - Search transcript content
   - Date range selection

2. **Statistics Dashboard**
   - Total summaries
   - Total digital signals detected
   - Most active callsigns
   - Signal type distribution chart

3. **Export Functionality**
   - Export to JSON
   - Export to CSV
   - Download individual summaries

4. **Live Monitoring**
   - WebSocket for real-time updates
   - Show current batch progress
   - Display "listening" status

---

## 📁 Key Files & Locations

**Core Code:**
- `bot_passive.py:1-184` - Main bot server (add signal detector here)
- `passive_processors.py:1-337` - Batch processor, summary writer, Pydantic models
- `summary_viewer.py:1-403` - Web UI server
- `signal_detector.py` - NEW: To be created

**Documentation:**
- `docs/claude/planning/DIGITAL-SIGNAL-DETECTION.md` - Implementation guide
- `NEXT_SESSION_PROMPT.md` - This file

**Test Files:**
- `test_passive_integration.py` - Integration tests
- `recordings/*.wav` - Real radio recordings for testing

**Data Directories:**
- `summaries/` - Generated summary files (4 files currently)
- `recordings/` - Test audio recordings (5 WAV files)
- `test_summaries/` - Integration test output

**Web UI:**
- URL: http://localhost:8080
- Port: 8080 (configurable)

---

## 🔍 Investigation Questions

**Answer these during the session:**

1. **What's the false positive rate for digital signal detection?**
   - Test with voice-only recordings
   - Test with known digital signals
   - Tune thresholds to minimize false positives

2. **Can we distinguish between different digital signal types?**
   - FT8 vs RTTY vs PSK31
   - Use spectral features (peak frequency, bandwidth)
   - May need ML for accurate classification

3. **How does signal detection affect performance?**
   - Does FFT analysis slow down the pipeline?
   - Should we downsample audio before analysis?
   - Can we run detection in parallel?

4. **Are the existing summaries accurate?**
   - Review summaries in web UI
   - Do topics make sense?
   - Are callsigns being detected correctly?
   - Is the LLM summary helpful?

---

## 🚀 Success Criteria

**By end of session, should have:**

✅ **Digital Signal Detection Working**
- [ ] SignalDetector processor implemented
- [ ] DigitalSignalFrame created
- [ ] Wired into pipeline and batch processor
- [ ] Tested with real recordings
- [ ] Digital signals appearing in web UI

✅ **Detection Quality Validated**
- [ ] False positive rate < 5% on voice-only recordings
- [ ] Detects known digital signals (FT8, RTTY, etc.)
- [ ] Timestamps and durations are accurate
- [ ] Signal types classified (even if basic)

✅ **Documentation Updated**
- [ ] Updated DIGITAL-SIGNAL-DETECTION.md with findings
- [ ] Noted any challenges or gotchas
- [ ] Updated this file for next session

---

## 💡 Tips & Reminders

**Audio Analysis:**
- Use `numpy` for FFT and signal processing
- Consider downsampling to 8kHz for faster processing
- Cache FFT results if analyzing same audio twice
- Digital signals typically have flat spectrum (high entropy)

**Pipeline Integration:**
- Audio frames pass through processors automatically
- Don't consume AudioRawFrames (let them flow through)
- Emit custom frames (DigitalSignalFrame) alongside audio
- Use event handlers to collect custom frames

**Testing:**
- Start with `test2.wav` (quick, known content)
- Then test with `recordings/merged_*.wav` (real radio)
- Use web UI to verify results visually
- Check logs for detection events

**Common Pitfalls:**
- FFT on 16kHz audio can be slow - consider downsampling
- Too sensitive threshold = many false positives
- Too high threshold = miss real signals
- Speaker detection still crude (pause-based) - many false speakers

---

## 📚 References

**Audio Signal Processing:**
- NumPy FFT: https://numpy.org/doc/stable/reference/routines.fft.html
- Spectral flatness: https://en.wikipedia.org/wiki/Spectral_flatness
- Digital modulation: https://www.sigidwiki.com/

**Pipecat:**
- FrameProcessor: `.venv/lib/python3.12/site-packages/pipecat/processors/frame_processor.py`
- Custom frames: `.venv/lib/python3.12/site-packages/pipecat/frames/frames.py`
- Event handlers: https://github.com/pipecat-ai/pipecat/tree/main/examples

**Digital Signal Samples:**
- FT8: https://physics.princeton.edu/pulsar/k1jt/FT8_samples.wav
- RTTY: https://www.sigidwiki.com/wiki/Radioteletype
- PSK31: https://www.sigidwiki.com/wiki/PSK31

Good luck! 🚀
