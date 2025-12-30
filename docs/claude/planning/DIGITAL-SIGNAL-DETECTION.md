# Digital Signal Detection - Future Implementation

**Created:** 2025-12-30
**Status:** Planning - Not Yet Implemented
**Related Files:** `passive_processors.py:27-31`, `bot_passive.py:56-71`

## Current State

The passive monitoring bot now supports **logging digital signals** in summary files via the `DigitalSignal` Pydantic model:

```python
class DigitalSignal(BaseModel):
    timestamp: str
    duration_seconds: float
    signal_type: str | None  # e.g., "FT8", "RTTY", "PSK31"
```

However, **no actual detection is implemented yet**. The LLM can include digital signals in its response, but the bot doesn't analyze audio for digital signal patterns.

## Architecture: Audio Routing to Multiple Processors

Pipecat supports sending the same audio stream to multiple processors simultaneously. Three approaches:

### Option 1: Sequential Processing (Simplest)

Audio frames automatically pass through processors that don't consume them:

```python
pipeline = Pipeline([
    transport.input(),
    whisper_stt,           # Consumes audio, outputs TranscriptionFrames
    signal_detector,       # ALSO gets audio frames (they pass through)
    transcript_processor,  # Gets transcription frames
    ...
])
```

**Location:** `bot_passive.py:101-111`
**Implementation:** Add `signal_detector` after `stt` in the pipeline

### Option 2: Parallel Pipeline Branches

For truly independent processing paths:

```python
from pipecat.pipeline.parallel_pipeline import ParallelPipeline

pipeline = Pipeline([
    transport.input(),
    ParallelPipeline([
        [whisper_stt, transcript_processor],      # Voice path
        [signal_detector, digital_signal_logger], # Digital signal path
    ]),
    # Both paths merge here
    context_aggregator.user(),
    ...
])
```

**Use case:** When both processors need to run concurrently without blocking

### Option 3: Custom AudioTee Processor

For explicit frame cloning:

```python
class AudioTee(FrameProcessor):
    """Duplicate audio frames to multiple downstream processors."""

    def __init__(self, destinations: list[FrameProcessor]):
        super().__init__()
        self._destinations = destinations

    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            # Send copy to each destination
            for dest in self._destinations:
                await dest.process_frame(frame.copy(), direction)

        # Continue normal flow
        await self.push_frame(frame, direction)
```

**Use case:** Fine-grained control over which frames go where

## Proposed Signal Detector Implementation

### Basic Amplitude-Based Detector

```python
class SignalDetector(FrameProcessor):
    """Detect digital signals based on audio characteristics.

    Digital signals typically have:
    - Sustained tones (FSK, AFSK)
    - Rapid amplitude changes (PSK)
    - Distinct spectral patterns
    """

    def __init__(self, threshold_db=-20, min_duration_sec=0.5):
        super().__init__()
        self._threshold = threshold_db
        self._min_duration = min_duration_sec

        # State tracking
        self._signal_start = None
        self._current_signal_type = None

    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            # Calculate audio properties
            amplitude_db = self._calculate_amplitude_db(frame.audio)
            spectral_features = self._analyze_spectrum(frame.audio)

            # Detect digital signal patterns
            is_digital = self._is_digital_signal(amplitude_db, spectral_features)

            if is_digital and self._signal_start is None:
                # Signal started
                self._signal_start = datetime.now()
                self._current_signal_type = self._classify_signal(spectral_features)
                logger.debug(f"Digital signal started: {self._current_signal_type}")

            elif not is_digital and self._signal_start is not None:
                # Signal ended
                duration = (datetime.now() - self._signal_start).total_seconds()

                if duration >= self._min_duration:
                    # Emit DigitalSignalFrame
                    await self.push_frame(
                        DigitalSignalFrame(
                            timestamp=self._signal_start.isoformat(),
                            duration_seconds=duration,
                            signal_type=self._current_signal_type
                        ),
                        direction
                    )

                self._signal_start = None
                self._current_signal_type = None

        # Pass audio through
        await self.push_frame(frame, direction)

    def _calculate_amplitude_db(self, audio: bytes) -> float:
        """Calculate RMS amplitude in dB."""
        import numpy as np
        samples = np.frombuffer(audio, dtype=np.int16)
        rms = np.sqrt(np.mean(samples**2))
        return 20 * np.log10(rms + 1e-10)

    def _analyze_spectrum(self, audio: bytes) -> dict:
        """Analyze frequency spectrum using FFT."""
        import numpy as np
        samples = np.frombuffer(audio, dtype=np.int16)
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)

        return {
            "peak_frequency": np.argmax(magnitude),
            "spectral_flatness": self._spectral_flatness(magnitude),
            "harmonic_ratio": self._harmonic_ratio(magnitude)
        }

    def _is_digital_signal(self, amplitude_db: float, spectral: dict) -> bool:
        """Determine if audio contains digital signal.

        Digital signals typically have:
        - High spectral flatness (noise-like)
        - Low harmonic content
        - Sustained amplitude
        """
        return (
            amplitude_db > self._threshold and
            spectral["spectral_flatness"] > 0.5 and
            spectral["harmonic_ratio"] < 0.3
        )

    def _classify_signal(self, spectral: dict) -> str:
        """Attempt to classify signal type based on spectral features."""
        # Very basic classification - would need ML for accuracy
        peak_freq = spectral["peak_frequency"]

        if 1000 < peak_freq < 2000:
            return "RTTY"  # Typical RTTY frequencies
        elif 500 < peak_freq < 1500:
            return "PSK31"  # PSK31 common range
        else:
            return "Unknown Digital"
```

### Advanced: TorchSig Integration

For more sophisticated signal classification:

```python
import torchsig

class TorchSigDetector(FrameProcessor):
    """Use TorchSig for ML-based signal classification."""

    def __init__(self):
        super().__init__()
        # Load pre-trained model
        self._model = torchsig.models.load_pretrained("signal_classifier")

    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            # Convert audio to IQ samples
            iq_samples = self._audio_to_iq(frame.audio)

            # Classify signal
            prediction = self._model.predict(iq_samples)

            if prediction.modulation_type != "voice":
                # Detected digital modulation
                await self.push_frame(
                    DigitalSignalFrame(
                        timestamp=datetime.now().isoformat(),
                        duration_seconds=frame.duration,
                        signal_type=prediction.modulation_type
                    ),
                    direction
                )

        await self.push_frame(frame, direction)
```

## Integration with Summary Writer

The `SummaryWriter` already supports digital signals - just needs to receive them:

### Create DigitalSignalFrame

```python
# In pipecat/frames/frames.py (or local custom_frames.py)
class DigitalSignalFrame(DataFrame):
    """Frame containing detected digital signal information."""

    def __init__(
        self,
        timestamp: str,
        duration_seconds: float,
        signal_type: str | None = None
    ):
        super().__init__()
        self.timestamp = timestamp
        self.duration_seconds = duration_seconds
        self.signal_type = signal_type
```

### Collect in TranscriptBatchProcessor

```python
# In passive_processors.py:42
class TranscriptBatchProcessor:
    def __init__(self, ...):
        ...
        self._digital_signals = []  # Add this

    async def add_digital_signal(self, frame: DigitalSignalFrame):
        """Collect digital signal detections."""
        self._digital_signals.append({
            "timestamp": frame.timestamp,
            "duration_seconds": frame.duration_seconds,
            "signal_type": frame.signal_type
        })

    async def _process_batch(self):
        ...
        # Include digital signals in metadata
        self._summary_writer._pending_batch_metadata = {
            ...
            "digital_signals": self._digital_signals.copy()
        }
        self._digital_signals = []  # Reset for next batch
```

### Wire Up Event Handler

```python
# In bot_passive.py after line 135
@signal_detector.event_handler("on_digital_signal_detected")
async def on_digital_signal(detector, frame):
    await batch_processor.add_digital_signal(frame)
```

## Implementation Checklist

- [ ] Choose signal detection approach (amplitude-based vs TorchSig)
- [ ] Implement `SignalDetector` processor
- [ ] Create `DigitalSignalFrame` class
- [ ] Add digital signal collection to `TranscriptBatchProcessor`
- [ ] Wire up event handler in `bot_passive.py`
- [ ] Test with recorded digital signals (FT8, RTTY, PSK31)
- [ ] Tune detection thresholds for false positive rate
- [ ] Add signal type classification (if using ML)

## Testing Resources

- **FT8 samples:** https://physics.princeton.edu/pulsar/k1jt/FT8_samples.wav
- **RTTY samples:** https://www.sigidwiki.com/wiki/Radioteletype
- **Test recordings:** Place in `recordings/` directory

## References

- Pipecat FrameProcessor: `.venv/lib/python3.12/site-packages/pipecat/processors/frame_processor.py`
- Pipeline docs: https://github.com/pipecat-ai/pipecat/tree/main/examples
- TorchSig: https://github.com/TorchDSP/torchsig
- Digital modes reference: https://www.sigidwiki.com/
