# Architecture Analysis & Recommendations

**Date:** 2025-12-30
**Status:** Planning & Discussion
**Context:** Critical analysis of tech stack and architecture for reactive radio monitoring system

---

## Project Goals (Clarified)

**NOT a passive logger** - this is a **reactive event-driven monitoring system**:

### Core Requirements
1. **Multi-path audio analysis**
   - Text transcription (Whisper)
   - Emotion/tone detection (audio models)
   - Digital signal detection (FFT)
   - Voice recognition (family members)
   - Frequency analysis

2. **Event-driven reactions**
   - Notifications for topics of interest
   - Route digital signals to appropriate decoders
   - Alert on family voice detection
   - Enable control mode for authorized speakers
   - Future: auto-response as dispatch agent

3. **Logging & summaries**
   - Searchable logs of communications
   - Summary generation (LLM-based)
   - Audio archival for playback/verification

4. **Future bidirectional capability**
   - Auto dispatch agent (respond to queries)
   - Two-way radio control
   - Automated responses

---

## ✅ Current Architecture is Sound

### Pipecat Framework Choice: CORRECT

**Original critique was wrong.** Pipecat is appropriate because:

1. **Event-driven architecture**
   - Frame-based processing with event handlers
   - Perfect for "when X detected, do Y" logic
   - Async/concurrent by design

2. **Multi-path audio routing**
   - Can fork audio to multiple processors
   - Parallel analysis without blocking
   - Easy to add new analysis paths

3. **LLM integration built-in**
   - Already working with Ollama
   - Needed for topic detection, summaries, alerts
   - Future: dispatch agent conversation

4. **Extensible processor model**
   - Custom frames: `DigitalSignalFrame`, `EmotionFrame`, `FamilyVoiceFrame`
   - Easy to add new analyzers
   - Clean separation of concerns

5. **Future bidirectional ready**
   - Can add TTS when needed
   - Response generation already supported
   - Conversation management built-in

**Verdict:** Pipecat is the right tool for a reactive audio monitoring system.

### Alternative: GNU Radio
- **Pros:** Powerful DSP, visual programming, hardware integration
- **Cons:** Complex for ML/LLM, C++ learning curve, overkill for this
- **Use case:** If you needed low-level SDR signal processing (you don't currently)

### Alternative: Roll Your Own
- **Pros:** Full control, minimal dependencies
- **Cons:** Rebuild event system, multi-path routing, LLM integration from scratch
- **Verdict:** Not worth it when Pipecat already provides this

---

## 🗄️ Storage Strategy: The Database Question

### Current: Text Files
```
summaries/20251230-120956_antenna_testing.txt
```

**Pros:**
- ✅ Simple, no database to manage
- ✅ Human-readable (can `cat` or `grep` files)
- ✅ Easy backup (just copy directory)
- ✅ Append-only, no corruption on crash
- ✅ Can scale horizontally (shard by date/frequency)
- ✅ Works with standard Unix tools

**Cons:**
- ❌ No indexing - can't efficiently query "all W1ABC contacts"
- ❌ No relationships - can't link conversations or track patterns
- ❌ Parsing required - must parse text file to extract metadata
- ❌ No atomic updates - can't update summary without rewriting file
- ❌ No aggregations - can't easily answer "most active callsigns this month"

### Option A: Keep Text Files + Add Indexes

**Approach:** Files for storage, lightweight index for queries

```python
# Still write text files
summaries/20251230-120956_antenna_testing.txt

# But also maintain JSON index
index.jsonl  # Newline-delimited JSON
{
  "timestamp": "2025-12-30T12:09:56Z",
  "file": "20251230-120956_antenna_testing.txt",
  "callsigns": ["W1ABC", "K2XYZ"],
  "topics": ["antenna testing", "signal report"],
  "duration": 300,
  "has_audio": true
}
{...next entry...}
```

**Query via:**
```bash
# Find all W1ABC contacts
jq 'select(.callsigns[] == "W1ABC")' index.jsonl

# Most active callsigns this month
jq -s '[.[].callsigns[]] | group_by(.) | map({callsign: .[0], count: length})' index.jsonl
```

**Pros:**
- ✅ Keep simple file-based storage
- ✅ Fast queries via index (in-memory JSON)
- ✅ Can rebuild index from files if corrupted
- ✅ Easy backup (files + index)

**Cons:**
- ⚠️ Index can get out of sync with files
- ⚠️ Have to maintain index consistency
- ⚠️ Aggregations still require processing full index

### Option B: Document Database (MongoDB, CouchDB)

**Approach:** JSON documents with indexing

```javascript
// MongoDB collection: summaries
{
  "_id": ObjectId("..."),
  "timestamp": ISODate("2025-12-30T12:09:56Z"),
  "session_start": ISODate("2025-12-30T12:09:56Z"),
  "session_end": ISODate("2025-12-30T12:09:57Z"),
  "duration": 300,
  "callsigns": ["W1ABC", "K2XYZ"],
  "topics": ["antenna testing", "signal report"],
  "transcript": [
    {"time": "12:09:56", "speaker": 1, "text": "This is W1ABC..."},
    ...
  ],
  "summary": "W1ABC and K2XYZ are discussing...",
  "digital_signals": [],
  "audio_file": "recordings/2025-12-30/120956.wav"
}

// Indexes
db.summaries.createIndex({"callsigns": 1})
db.summaries.createIndex({"timestamp": -1})
db.summaries.createIndex({"topics": 1})
```

**Queries:**
```javascript
// All W1ABC contacts
db.summaries.find({"callsigns": "W1ABC"})

// Most active callsigns
db.summaries.aggregate([
  {$unwind: "$callsigns"},
  {$group: {_id: "$callsigns", count: {$sum: 1}}},
  {$sort: {count: -1}}
])
```

**Pros:**
- ✅ Flexible schema (JSON documents)
- ✅ Indexed queries (fast searches)
- ✅ Aggregation pipeline (analytics)
- ✅ Still stores JSON (not relational overhead)
- ✅ Can export to files for backup

**Cons:**
- ⚠️ Another service to run (MongoDB/CouchDB)
- ⚠️ More complex than files
- ⚠️ Not human-readable (need tools to query)

### Option C: SQLite (Embedded SQL)

**Approach:** Lightweight SQL database (single file)

```sql
-- schema.sql
CREATE TABLE summaries (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    session_start DATETIME,
    session_end DATETIME,
    duration INTEGER,
    summary TEXT,
    transcript JSON,  -- SQLite 3.38+ supports JSON
    audio_file TEXT
);

CREATE TABLE callsigns (
    summary_id INTEGER REFERENCES summaries(id),
    callsign TEXT,
    PRIMARY KEY (summary_id, callsign)
);
CREATE INDEX idx_callsign ON callsigns(callsign);

CREATE TABLE topics (
    summary_id INTEGER REFERENCES summaries(id),
    topic TEXT
);
CREATE INDEX idx_topic ON topics(topic);
```

**Queries:**
```sql
-- All W1ABC contacts
SELECT s.* FROM summaries s
JOIN callsigns c ON s.id = c.summary_id
WHERE c.callsign = 'W1ABC'
ORDER BY s.timestamp DESC;

-- Most active callsigns
SELECT callsign, COUNT(*) as contacts
FROM callsigns
GROUP BY callsign
ORDER BY contacts DESC
LIMIT 10;
```

**Pros:**
- ✅ No separate database service (embedded)
- ✅ Single file: `radio_logs.db`
- ✅ Fast indexed queries
- ✅ SQL for complex analytics
- ✅ Transaction safety (ACID)
- ✅ Can export to CSV/JSON easily

**Cons:**
- ⚠️ Not human-readable (need sqlite3 tool)
- ⚠️ Schema migrations needed for changes
- ⚠️ Relational model (more rigid than document DB)

### Option D: Hybrid: Files + SQLite

**Best of both worlds:**

```python
# Store full summaries as text files (human-readable archive)
summaries/20251230-120956_antenna_testing.txt

# Store metadata in SQLite (fast queries)
radio_logs.db:
  - summaries table (minimal: id, timestamp, file_path, duration)
  - callsigns table (for searching)
  - topics table (for filtering)
  - audio_files table (references to recordings)
```

**Workflow:**
```python
async def save_summary(summary: RadioSummary, transcript: list, metadata: dict):
    # 1. Write human-readable text file (archive)
    text_file = f"summaries/{timestamp}_{slug}.txt"
    await write_summary_file(text_file, summary, transcript)

    # 2. Store metadata in SQLite (queries)
    db.execute("""
        INSERT INTO summaries (timestamp, file_path, duration, summary_text)
        VALUES (?, ?, ?, ?)
    """, (timestamp, text_file, duration, summary.summary))

    summary_id = db.lastrowid

    # 3. Store callsigns (indexed)
    for callsign in summary.callsigns:
        db.execute("INSERT INTO callsigns VALUES (?, ?)", (summary_id, callsign))

    # 4. Store topics (indexed)
    for topic in summary.topics:
        db.execute("INSERT INTO topics VALUES (?, ?)", (summary_id, topic))
```

**Query:**
```python
# Fast query via SQLite
results = db.execute("""
    SELECT s.timestamp, s.file_path, s.summary_text
    FROM summaries s
    JOIN callsigns c ON s.id = c.summary_id
    WHERE c.callsign = 'W1ABC'
    ORDER BY s.timestamp DESC
""").fetchall()

# Read full transcript from file if needed
for row in results:
    full_content = open(row['file_path']).read()
```

**Pros:**
- ✅ Human-readable archive (text files)
- ✅ Fast queries (SQLite indexes)
- ✅ Best of both worlds
- ✅ Can regenerate DB from files if needed
- ✅ Simple backup (copy directory + .db file)

**Cons:**
- ⚠️ Maintain two stores (files + DB)
- ⚠️ Can get out of sync (but recoverable)

---

## 📊 Storage Recommendation Matrix

| Use Case | Recommended Approach |
|----------|---------------------|
| **Just logging, no queries** | Text files only (current) |
| **Need simple search** | Text files + JSON index |
| **Need analytics/aggregations** | Hybrid (files + SQLite) |
| **Flexible schema, document-oriented** | MongoDB/CouchDB |
| **Complex relational queries** | PostgreSQL |

### My Recommendation: **Hybrid (Files + SQLite)**

**Rationale:**
1. Keep text files for human-readable archive (grep, cat, backup)
2. Add SQLite for indexed queries (callsign search, date filtering)
3. Single file DB (`radio_logs.db`) - no service to manage
4. Can regenerate DB from files if corrupted
5. Easy to add later - doesn't break current setup

**Implementation:**
```python
# 1. Keep existing summary_writer.py writing text files
# 2. Add DatabaseWriter processor:

class DatabaseWriter(FrameProcessor):
    def __init__(self, db_path: str = "radio_logs.db"):
        super().__init__()
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    async def process_frame(self, frame: Frame, direction):
        # Listen for summary completion events
        if isinstance(frame, SummaryCompleteFrame):
            await self._store_metadata(frame)

        await self.push_frame(frame, direction)
```

---

## 🎯 Multi-Path Audio Analysis Architecture

### Proposed Pipeline Structure

```python
# bot_passive.py - Enhanced architecture

pipeline = Pipeline([
    transport.input(),

    # Audio Tee - fork to multiple analysis paths
    AudioTee([
        # PATH 1: Speech → Text → LLM → Summaries + Alerts
        [
            whisper_stt,
            transcript_processor.user(),
            context_aggregator.user(),
            llm,
            summary_writer,
            database_writer,           # NEW: Store metadata
            alert_processor,           # NEW: Topic-based alerts
            context_aggregator.assistant()
        ],

        # PATH 2: Audio → Emotion/Tone Detection
        [
            audio_emotion_detector,     # NEW: Detect singing, stress, urgency
            emotion_alert_processor     # NEW: Alert on emotional cues
        ],

        # PATH 3: Digital Signal Detection
        [
            signal_detector,            # NEW: FFT-based detection
            signal_router,              # NEW: Route to decoders
            # Future: ft8_decoder, rtty_decoder, psk31_decoder
        ],

        # PATH 4: Voice Recognition
        [
            voice_embedding_extractor,  # NEW: Speaker embeddings
            family_detector,            # NEW: Match against family profiles
            family_alert_processor      # NEW: Alert + enable control mode
        ],

        # PATH 5: Audio Archival
        [
            audio_file_writer          # NEW: Save WAV files
        ]
    ])
])
```

### Event Handler Examples

```python
# Topic-based alerts
@alert_processor.event_handler("on_topic_detected")
async def on_topic(processor, frame: TopicFrame):
    if frame.topic in ["emergency", "fire", "medical"]:
        await send_notification(
            title="🚨 Emergency Traffic Detected",
            message=f"Topic: {frame.topic}",
            priority="high"
        )

# Family voice detection
@family_detector.event_handler("on_family_voice")
async def on_family(processor, frame: FamilyVoiceFrame):
    await send_notification(
        title=f"👤 {frame.person} on frequency",
        message=f"Confidence: {frame.confidence:.2%}",
        priority="normal"
    )

    # Enable control mode for authorized users
    if frame.person in AUTHORIZED_USERS:
        await enable_control_mode(frame.person)

# Digital signal routing
@signal_router.event_handler("on_signal_classified")
async def route_signal(processor, frame: DigitalSignalFrame):
    if frame.signal_type == "FT8":
        await ft8_decoder.process(frame)
    elif frame.signal_type == "RTTY":
        await rtty_decoder.process(frame)
```

---

## 🔧 Implementation Roadmap

### Phase 1: Foundation (Current + DB) - 2-3 days
- [x] Text transcription + summaries (DONE)
- [x] JSON parsing (DONE)
- [x] Web UI (DONE)
- [ ] Add SQLite database writer
- [ ] Add audio file archival
- [ ] Update web UI to query from DB

### Phase 2: Multi-Path Analysis - 1 week
- [ ] Implement AudioTee processor
- [ ] Add digital signal detector (FFT-based)
- [ ] Add audio emotion detector
- [ ] Add voice embedding extractor
- [ ] Wire up parallel paths

### Phase 3: Reactive Behaviors - 1 week
- [ ] Notification service (Pushover/Telegram/Signal)
- [ ] Alert processor (topic-based triggers)
- [ ] Family voice detector + alert
- [ ] Signal router (digital signal → decoder)
- [ ] Control mode (authorized voice → commands)

### Phase 4: Advanced Features - 2 weeks
- [ ] FT8 decoder integration
- [ ] RTTY decoder integration
- [ ] Audio analytics (spectrograms, waterfalls)
- [ ] Contact logging (ADIF export)
- [ ] Web UI enhancements (live monitoring, search)

### Phase 5: Active Dispatch Agent (Future)
- [ ] Add TTS for responses
- [ ] LLM-based dispatch logic
- [ ] Two-way radio control
- [ ] Automated responses to queries

---

## 📝 Key Decisions Summary

### ✅ Confirmed Good Choices
1. **Pipecat framework** - Right tool for reactive audio monitoring
2. **Whisper STT** - Excellent accuracy, works offline
3. **Ollama (local LLM)** - Privacy, no API costs, needed for summaries + dispatch
4. **FastAPI** - Modern web framework, good choice
5. **Python** - Best ecosystem for ML/audio/DSP

### 🔄 Recommended Changes
1. **Storage:** Add SQLite alongside text files (hybrid approach)
   - Keep human-readable archive
   - Add indexed queries
   - Can regenerate DB from files

2. **Audio Input:** Add direct audio input option
   - Keep WebRTC for testing/demo
   - Add sounddevice/PyAudio for 24/7 operation
   - Support both modes

3. **Speaker Detection:** Accept limitation or use voice embeddings
   - Current pause-based detection is misleading (50 speakers)
   - Either: Use SpeechBrain embeddings, OR
   - Label as "Utterances" not "Speakers"

4. **Architecture:** Multi-path audio routing
   - Implement AudioTee processor
   - Add parallel analysis paths
   - Event-driven reactions

---

## 🎓 Lessons Learned

1. **Always understand the full use case before critiquing**
   - Initial analysis missed "reactive monitoring" vs "passive logging"
   - Pipecat is correct choice for event-driven reactions

2. **Files vs DB is not binary**
   - Hybrid approach gets best of both
   - Text files: human-readable archive
   - SQLite: indexed queries
   - Can choose based on actual query needs

3. **Future-proofing matters**
   - Pipecat's bidirectional capability will be needed (dispatch agent)
   - Better to use framework that supports future requirements
   - Even if slightly heavier now

4. **User knows their domain**
   - Radio monitoring has specific needs (ADIF export, audio archival)
   - Listen to domain expertise before suggesting generic solutions

---

## 🚀 Next Steps

**Immediate (This Week):**
1. Add SQLite database writer (keep text files too)
2. Add audio file archival (save WAV alongside transcripts)
3. Update web UI to show "Utterances" not "Speakers" (until we have real speaker detection)

**Short-term (Next 2 Weeks):**
1. Implement AudioTee for multi-path routing
2. Add digital signal detector (FFT-based)
3. Wire up notification system for topic alerts

**Medium-term (Next Month):**
1. Add voice recognition for family detection
2. Implement signal routing to decoders
3. Build out web UI search and analytics

**Long-term (Future):**
1. Active dispatch agent capability
2. Two-way radio control
3. Automated response system

---

## 📚 References

**Pipecat:**
- Framework docs: https://github.com/pipecat-ai/pipecat
- Custom processors: `.venv/lib/python3.12/site-packages/pipecat/processors/`
- Event handlers: See examples in repo

**Audio Analysis:**
- Emotion detection: https://huggingface.co/superb/wav2vec2-base-superb-er
- Speaker recognition: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- Digital signal processing: NumPy FFT docs

**Database:**
- SQLite JSON support: https://www.sqlite.org/json1.html
- SQLAlchemy: https://www.sqlalchemy.org/
- MongoDB (if going document DB): https://www.mongodb.com/docs/drivers/python/

**Radio Standards:**
- ADIF format: https://adif.org/
- Ham radio modes: https://www.sigidwiki.com/
- FT8 protocol: https://physics.princeton.edu/pulsar/k1jt/wsjtx.html
