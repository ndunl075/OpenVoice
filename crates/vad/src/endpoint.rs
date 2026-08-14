//! Endpointing state machine, kept independent of the ONNX inference call
//! so it's unit-testable with synthetic probability streams.
//!
//! See `dictation-architecture.md` §2.2: VAD has two jobs -- (a) trim
//! silence so the encoder never wastes compute on dead air, (b) detect
//! end-of-speech in ~2-3 frames instead of a fixed timeout. This module is
//! job (b) (and half of (a): it's what tells the caller which frames *are*
//! silence to trim).
//!
//! Debouncing matters here: a single frame flickering across the
//! probability threshold shouldn't start or end a speech segment on its
//! own -- a stray high-pitched vowel or a brief pause between words would
//! chop utterances into pieces otherwise. Requiring a run of consecutive
//! frames on either side of the transition is what "~2-3 frames" in the
//! doc actually buys.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointEvent {
    /// Enough consecutive frames crossed the speech threshold to confirm
    /// this is the start of an utterance, not noise.
    SpeechStart,
    /// Enough consecutive frames dropped back below threshold to confirm
    /// the utterance actually ended, not just a mid-sentence pause.
    SpeechEnd,
}

#[derive(Debug, Clone, Copy)]
pub struct EndpointConfig {
    /// Model output above this probability counts as "speech" for one frame.
    pub speech_threshold: f32,
    /// Consecutive speech-frames required to confirm SpeechStart.
    pub start_frames: u32,
    /// Consecutive silence-frames required to confirm SpeechEnd.
    pub end_frames: u32,
}

impl Default for EndpointConfig {
    /// §2.3 frames the whole point as "detect end-of-speech in ~2-3 frames
    /// instead of a fixed timeout" -- 3 frames on either edge is the
    /// literal reading of that.
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            start_frames: 3,
            end_frames: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silence,
    /// Seen `count` consecutive speech frames, haven't confirmed start yet.
    MaybeSpeechStart { count: u32 },
    Speech,
    /// Seen `count` consecutive silence frames since confirmed speech,
    /// haven't confirmed end yet.
    MaybeSpeechEnd { count: u32 },
}

pub struct Endpointer {
    config: EndpointConfig,
    state: State,
}

impl Endpointer {
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            config,
            state: State::Silence,
        }
    }

    /// Whether the endpointer currently considers itself inside a speech
    /// segment (confirmed start, not yet confirmed end). Frames while this
    /// is false are the silence §2.2 says to trim before it ever reaches
    /// the encoder.
    pub fn in_speech(&self) -> bool {
        matches!(self.state, State::Speech | State::MaybeSpeechEnd { .. })
    }

    /// Resets to the initial silent state -- call between independent
    /// recording sessions so leftover debounce counters from a previous
    /// utterance don't affect the next one.
    pub fn reset(&mut self) {
        self.state = State::Silence;
    }

    /// Feeds one frame's speech probability (as returned by
    /// [`crate::SileroVad::process_frame`]). Returns an event on confirmed
    /// transitions only.
    pub fn push_probability(&mut self, probability: f32) -> Option<EndpointEvent> {
        let is_speech_frame = probability >= self.config.speech_threshold;

        self.state = match (self.state, is_speech_frame) {
            (State::Silence, false) => State::Silence,
            (State::Silence, true) => State::MaybeSpeechStart { count: 1 },

            (State::MaybeSpeechStart { count }, true) => {
                State::MaybeSpeechStart { count: count + 1 }
            }
            (State::MaybeSpeechStart { .. }, false) => State::Silence,

            (State::Speech, true) => State::Speech,
            (State::Speech, false) => State::MaybeSpeechEnd { count: 1 },

            (State::MaybeSpeechEnd { count }, false) => State::MaybeSpeechEnd { count: count + 1 },
            (State::MaybeSpeechEnd { .. }, true) => State::Speech,
        };

        match self.state {
            State::MaybeSpeechStart { count } if count >= self.config.start_frames => {
                self.state = State::Speech;
                Some(EndpointEvent::SpeechStart)
            }
            State::MaybeSpeechEnd { count } if count >= self.config.end_frames => {
                self.state = State::Silence;
                Some(EndpointEvent::SpeechEnd)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> EndpointConfig {
        EndpointConfig {
            speech_threshold: 0.5,
            start_frames: 3,
            end_frames: 3,
        }
    }

    #[test]
    fn silence_never_fires() {
        let mut e = Endpointer::new(config());
        for _ in 0..20 {
            assert_eq!(e.push_probability(0.1), None);
        }
        assert!(!e.in_speech());
    }

    #[test]
    fn sub_threshold_run_never_fires() {
        // Below start_frames, even if every frame is speech-like.
        let mut e = Endpointer::new(config());
        assert_eq!(e.push_probability(0.9), None);
        assert_eq!(e.push_probability(0.9), None);
        assert!(!e.in_speech());
    }

    #[test]
    fn confirms_speech_start_after_run() {
        let mut e = Endpointer::new(config());
        assert_eq!(e.push_probability(0.9), None);
        assert_eq!(e.push_probability(0.9), None);
        assert_eq!(e.push_probability(0.9), Some(EndpointEvent::SpeechStart));
        assert!(e.in_speech());
    }

    #[test]
    fn fires_speech_start_exactly_once() {
        let mut e = Endpointer::new(config());
        for _ in 0..3 {
            e.push_probability(0.9);
        }
        // Already confirmed; more speech frames shouldn't re-fire.
        for _ in 0..10 {
            assert_eq!(e.push_probability(0.9), None);
        }
    }

    #[test]
    fn brief_dip_does_not_end_speech() {
        let mut e = Endpointer::new(config());
        for _ in 0..3 {
            e.push_probability(0.9);
        }
        assert!(e.in_speech());
        // Two silent frames -- short of end_frames=3 -- then back to speech.
        assert_eq!(e.push_probability(0.1), None);
        assert_eq!(e.push_probability(0.1), None);
        assert!(e.in_speech(), "still counts as speech mid-debounce");
        assert_eq!(e.push_probability(0.9), None);
        assert!(e.in_speech(), "should recover, not end, on a brief dip");
    }

    #[test]
    fn confirms_speech_end_after_silence_run() {
        let mut e = Endpointer::new(config());
        for _ in 0..3 {
            e.push_probability(0.9);
        }
        assert_eq!(e.push_probability(0.1), None);
        assert_eq!(e.push_probability(0.1), None);
        assert_eq!(e.push_probability(0.1), Some(EndpointEvent::SpeechEnd));
        assert!(!e.in_speech());
    }

    #[test]
    fn supports_multiple_utterances_back_to_back() {
        let mut e = Endpointer::new(config());
        for _ in 0..2 {
            for _ in 0..2 {
                assert_eq!(e.push_probability(0.9), None);
            }
            assert_eq!(e.push_probability(0.9), Some(EndpointEvent::SpeechStart));
            for _ in 0..2 {
                assert_eq!(e.push_probability(0.1), None);
            }
            assert_eq!(e.push_probability(0.1), Some(EndpointEvent::SpeechEnd));
        }
    }

    #[test]
    fn reset_clears_in_progress_debounce() {
        let mut e = Endpointer::new(config());
        e.push_probability(0.9);
        e.push_probability(0.9);
        e.reset();
        assert!(!e.in_speech());
        // Needs a fresh full run to confirm start again.
        assert_eq!(e.push_probability(0.9), None);
        assert_eq!(e.push_probability(0.9), None);
        assert_eq!(e.push_probability(0.9), Some(EndpointEvent::SpeechStart));
    }

    #[test]
    fn probability_exactly_at_threshold_counts_as_speech() {
        let mut e = Endpointer::new(config());
        for _ in 0..3 {
            assert!(e.push_probability(0.5).is_none() || e.in_speech());
        }
        assert!(e.in_speech());
    }
}
