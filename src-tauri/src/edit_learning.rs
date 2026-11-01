//! Learns Custom Words from the user's own corrections.
//!
//! After a paste, the focused text field is read back through UI Automation
//! for a short window. When a transcribed word is replaced by a similar one
//! (e.g. "Ongoneyo" -> "Ongonyo"), the replacement is added to
//! `custom_words`, so fuzzy correction fixes it on the next transcription.
//! Field text never leaves memory or the log; only the learned word is stored.

use std::collections::HashSet;

use natural::phonetics::soundex;
use strsim::levenshtein;

use crate::audio_toolkit::text::{build_match_key, is_supported_fuzzy_key};

/// Most transcribed words one correction may replace ("Ongo neyo" -> "Ongonyo").
/// Matches the n-gram limit in `apply_custom_words`.
const MAX_REPLACED_WORDS: usize = 3;
/// Largest changed span (in words) worth diffing; bigger is a rewrite, not a fix.
const MAX_CHANGED_WORDS: usize = 64;
/// Normalized edit distance above which a replacement is a different word.
const MAX_EDIT_RATIO: f64 = 0.4;
/// Looser limit when both spellings sound alike ("Makari" -> "McCarrie").
const MAX_SOUND_ALIKE_EDIT_RATIO: f64 = 0.6;
/// Shortest learnable word; shorter ones would fuzzy-match too much.
const MIN_WORD_CHARS: usize = 4;

/// Words in `after` that correct words Handy pasted into `before`.
///
/// `pasted` is the transcript Handy inserted; `before` and `after` are the
/// whole field's text right after the paste and after the user's edits.
pub fn find_corrections(pasted: &str, before: &str, after: &str) -> Vec<String> {
    let old: Vec<&str> = before.split_whitespace().collect();
    let new: Vec<&str> = after.split_whitespace().collect();

    // Trim the shared prefix and suffix so a big field only diffs the edit.
    let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let old_mid = &old[prefix..old.len() - suffix];
    let new_mid = &new[prefix..new.len() - suffix];
    if old_mid.len() > MAX_CHANGED_WORDS || new_mid.len() > MAX_CHANGED_WORDS {
        return Vec::new();
    }

    let pasted_keys: HashSet<String> = pasted.split_whitespace().map(build_match_key).collect();
    let mut found = Vec::new();
    for hunk in replace_hunks(old_mid, new_mid) {
        let new_index = prefix + hunk.new_start;
        let previous = new_index.checked_sub(1).map(|i| new[i]);
        if let Some(word) = correction(&hunk.removed, hunk.added[0], previous, &pasted_keys) {
            if !found.contains(&word) {
                found.push(word);
            }
        }
    }
    found
}

struct Hunk<'a> {
    removed: Vec<&'a str>,
    added: Vec<&'a str>,
    new_start: usize,
}

/// Word-level LCS diff, keeping only hunks that swap 1-3 words for one word.
fn replace_hunks<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<Hunk<'a>> {
    let (n, m) = (old.len(), new.len());
    // lcs[i][j] = LCS length of old[i..] and new[j..].
    let mut lcs = vec![vec![0u16; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if old[i] == new[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }

    let mut hunks = Vec::new();
    let (mut i, mut j) = (0, 0);
    let mut current = Hunk {
        removed: Vec::new(),
        added: Vec::new(),
        new_start: 0,
    };
    let mut flush = |hunk: &mut Hunk<'a>, next_start: usize| {
        let done = std::mem::replace(
            hunk,
            Hunk {
                removed: Vec::new(),
                added: Vec::new(),
                new_start: next_start,
            },
        );
        if (1..=MAX_REPLACED_WORDS).contains(&done.removed.len()) && done.added.len() == 1 {
            hunks.push(done);
        }
    };
    while i < n || j < m {
        if i < n && j < m && old[i] == new[j] {
            flush(&mut current, j + 1);
            i += 1;
            j += 1;
        } else if j < m && (i == n || lcs[i][j + 1] >= lcs[i + 1][j]) {
            current.added.push(new[j]);
            j += 1;
        } else {
            current.removed.push(old[i]);
            i += 1;
        }
    }
    flush(&mut current, m);
    hunks
}

/// The learnable word if `added` is a fix of the pasted words in `removed`.
fn correction(
    removed: &[&str],
    added: &str,
    previous: Option<&str>,
    pasted_keys: &HashSet<String>,
) -> Option<String> {
    let word = added.trim_matches(|c: char| !c.is_alphanumeric());
    let new_key = build_match_key(word);
    let old_keys: Vec<String> = removed.iter().map(|w| build_match_key(w)).collect();
    if !old_keys.iter().all(|k| pasted_keys.contains(k)) {
        return None; // Edited text the user typed, not Handy's transcript.
    }
    let old_key = old_keys.concat();
    let is_change =
        removed.len() > 1 || removed[0].trim_matches(|c: char| !c.is_alphanumeric()) != word;
    // Growing or trimming a word ("profit" -> "profitable", "running" -> "run")
    // is rewording, not a misheard word.
    let is_reworded = old_key != new_key
        && (new_key.starts_with(&old_key)
            || old_key.starts_with(&new_key)
            || new_key.ends_with(&old_key)
            || old_key.ends_with(&new_key));
    if !is_change
        || is_reworded
        || !is_supported_fuzzy_key(&new_key)
        || new_key.chars().count() < MIN_WORD_CHARS
        || !looks_like_term(word, previous)
    {
        return None;
    }
    let max_len = old_key.chars().count().max(new_key.chars().count()) as f64;
    let ratio = levenshtein(&old_key, &new_key) as f64 / max_len;
    let sound_alike = [&old_key, &new_key]
        .iter()
        .all(|k| k.chars().all(|c| c.is_ascii_alphabetic()))
        && soundex(&old_key, &new_key);
    let limit = if sound_alike {
        MAX_SOUND_ALIKE_EDIT_RATIO
    } else {
        MAX_EDIT_RATIO
    };
    (ratio <= limit).then(|| word.to_string())
}

// shortcut: guesses "name or jargon" from shape (a digit, an inner or mid-sentence
// capital, or 6+ lowercase letters) so fixing "three" -> "There" can't teach a
// common word; names at a sentence start are missed. Upgrade with a word list.
fn looks_like_term(word: &str, previous: Option<&str>) -> bool {
    let mut chars = word.chars();
    let first_upper = chars.next().is_some_and(|c| c.is_ascii_uppercase());
    let inner_upper = chars.any(|c| c.is_ascii_uppercase());
    let sentence_start =
        previous.is_none_or(|p| p.ends_with(|c: char| matches!(c, '.' | '!' | '?' | ':')));
    word.chars().any(|c| c.is_ascii_digit())
        || inner_upper
        || (first_upper && !sentence_start)
        || (!first_upper && word.chars().count() >= 6)
}

#[cfg(target_os = "windows")]
pub use watch::watch_after_paste;

#[cfg(not(target_os = "windows"))]
pub fn watch_after_paste(_app: tauri::AppHandle, _pasted: String) {}

#[cfg(target_os = "windows")]
mod watch {
    use std::thread;
    use std::time::{Duration, Instant};

    use log::{debug, info, warn};
    use tauri::{AppHandle, Emitter};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
    };
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, IUIAutomation, IUIAutomationElement, IUIAutomationTextPattern,
        IUIAutomationValuePattern, UIA_TextPatternId, UIA_ValuePatternId,
    };

    use super::find_corrections;
    use crate::settings::{get_settings, write_settings};

    /// Stop watching once the field goes this long without a change, so a run
    /// of fixes keeps the watch open but walking away ends it.
    const IDLE_FOR: Duration = Duration::from_secs(10);
    /// Hard cap on how long after a paste edits are still attributed to it.
    const WATCH_AT_MOST: Duration = Duration::from_secs(60);
    const POLL_EVERY: Duration = Duration::from_millis(500);
    /// Reliable paste lands asynchronously; give the field this long to show it.
    const PASTE_ARRIVAL_TRIES: usize = 10;
    const PASTE_ARRIVAL_WAIT: Duration = Duration::from_millis(200);

    /// Watches the focused field after a paste and learns corrected words.
    pub fn watch_after_paste(app: AppHandle, pasted: String) {
        if !get_settings(&app).learn_from_edits || pasted.trim().is_empty() {
            return;
        }
        let spawned = thread::Builder::new()
            .name("edit-learning".into())
            .spawn(move || {
                let learned = watch(pasted.trim());
                if !learned.is_empty() {
                    add_custom_words(&app, learned);
                }
            });
        if let Err(e) = spawned {
            warn!("Failed to start the edit-learning thread: {e}");
        }
    }

    fn watch(pasted: &str) -> Vec<String> {
        let Some((field, before)) = (0..PASTE_ARRIVAL_TRIES).find_map(|_| {
            thread::sleep(PASTE_ARRIVAL_WAIT);
            let field = FocusedField::current()?;
            let text = field.text()?;
            text.contains(pasted).then_some((field, text))
        }) else {
            debug!("Edit learning: focused field doesn't expose the pasted text");
            return Vec::new();
        };

        // Judge only where the text finally settled (two polls in a row agree,
        // field not cleared by sending), so half-typed words and the in-between
        // states of a rewrite ("non-profit" -> "non-profitable" -> ...) never count.
        let started = Instant::now();
        let mut last_change = started;
        let mut last = before.clone();
        let mut settled: Option<String> = None;
        while last_change.elapsed() < IDLE_FOR
            && started.elapsed() < WATCH_AT_MOST
            && field.still_focused()
        {
            thread::sleep(POLL_EVERY);
            let Some(after) = field.text() else { break };
            if after != last {
                last_change = Instant::now();
            } else if !after.trim().is_empty() {
                settled = Some(after.clone());
            }
            last = after;
        }
        settled
            .map(|after| find_corrections(pasted, &before, &after))
            .unwrap_or_default()
    }

    fn add_custom_words(app: &AppHandle, words: Vec<String>) {
        let mut settings = get_settings(app);
        let new: Vec<String> = words
            .into_iter()
            .filter(|w| {
                !settings
                    .custom_words
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(w))
            })
            .collect();
        if new.is_empty() {
            return;
        }
        info!("Learned custom words from edits: {new:?}");
        settings.custom_words.extend(new.iter().cloned());
        write_settings(app, settings);
        let _ = app.emit("custom-words-learned", new);
    }

    /// The UI Automation element that had keyboard focus when it was captured.
    struct FocusedField {
        uia: IUIAutomation,
        element: IUIAutomationElement,
    }

    impl FocusedField {
        fn current() -> Option<Self> {
            unsafe {
                // Fine if COM is already initialized on this thread.
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                let uia: IUIAutomation =
                    CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).ok()?;
                let element = uia.GetFocusedElement().ok()?;
                if element.CurrentIsPassword().ok()?.as_bool() {
                    return None;
                }
                Some(Self { uia, element })
            }
        }

        fn still_focused(&self) -> bool {
            unsafe {
                self.uia
                    .GetFocusedElement()
                    .and_then(|focused| self.uia.CompareElements(&self.element, &focused))
                    .is_ok_and(|same| same.as_bool())
            }
        }

        fn text(&self) -> Option<String> {
            unsafe {
                if let Ok(pattern) = self
                    .element
                    .GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId)
                {
                    if let Ok(text) = pattern.DocumentRange().and_then(|r| r.GetText(-1)) {
                        return Some(text.to_string());
                    }
                }
                let pattern = self
                    .element
                    .GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId)
                    .ok()?;
                pattern.CurrentValue().ok().map(|v| v.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::find_corrections;

    #[test]
    fn learns_a_fixed_name() {
        let pasted = "I met Ongoneyo today";
        let before = "Notes: I met Ongoneyo today";
        let after = "Notes: I met Ongonyo today";
        assert_eq!(find_corrections(pasted, before, after), vec!["Ongonyo"]);
    }

    #[test]
    fn learns_words_merged_into_one() {
        let pasted = "call print F here";
        assert_eq!(
            find_corrections(pasted, pasted, "call printf here"),
            vec!["printf"]
        );
    }

    #[test]
    fn keeps_punctuation_out_of_the_word() {
        let pasted = "ask Ongoneyo, then leave";
        assert_eq!(
            find_corrections(pasted, pasted, "ask Ongonyo, then leave"),
            vec!["Ongonyo"]
        );
    }

    #[test]
    fn ignores_edits_outside_the_transcript() {
        let pasted = "hello there";
        let before = "Dear Ongoneyo, hello there";
        let after = "Dear Ongonyo, hello there";
        assert!(find_corrections(pasted, before, after).is_empty());
    }

    #[test]
    fn learns_several_fixes_at_once() {
        let pasted = "Felix arrived in the city of Baco to face the feared Bellbottom clan, \
                      the Fear Makari clan, and also Batharan.";
        let after = "Felix arrived in the city of Bako to face the feared Bellbottom clan, \
                     the Fear McCarrie clan, and also Betharan.";
        assert_eq!(
            find_corrections(pasted, pasted, after),
            vec!["Bako", "McCarrie", "Betharan"]
        );
    }

    #[test]
    fn ignores_rewording() {
        let pasted = "Truth Alliance is a non-profit";
        // The final rewrite, and a state passed through while typing it.
        for after in [
            "Truth Alliance is not very profitable",
            "Truth Alliance is a non-profitable",
            "Truth Alliance is a profit",
        ] {
            assert!(
                find_corrections(pasted, pasted, after).is_empty(),
                "{after}"
            );
        }
    }

    #[test]
    fn ignores_common_words_and_rewrites() {
        // Sentence-start capital and short words don't look like terms.
        assert!(find_corrections("three cats", "three cats", "There cats").is_empty());
        assert!(find_corrections("go to it", "go to it", "go too it").is_empty());
        // A different word is a rewrite, not a spelling fix.
        assert!(find_corrections("met Ongoneyo", "met Ongoneyo", "met Barbara").is_empty());
        // Retyping without a change, or deleting, learns nothing.
        assert!(find_corrections("met Ongoneyo", "met Ongoneyo", "met Ongoneyo").is_empty());
        assert!(find_corrections("met Ongoneyo", "met Ongoneyo", "").is_empty());
    }
}
