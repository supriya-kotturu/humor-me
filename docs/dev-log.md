# Development Log: Problems Hit

Every non-trivial problem met while building this project: what happened, why, how it was
caught, and the fix. Useful for the report's "challenges" section and for not repeating
mistakes. **Append new entries at the top of each section; never delete old ones.**

Format: **Symptom** → **Cause** → **Caught by** → **Fix** (commit / file).

---

## Data pipeline: extraction (S0)

### PDF has no paragraphs; every visual line is its own block
- **Symptom:** exact-match anchors ("Stand-Up Transcripts") never matched; text came out one line per block.
- **Cause:** browser "Save as PDF" emits one text block per printed line.
- **Caught by:** masked inspection of the first paragraphs.
- **Fix:** rebuild paragraphs from vertical gaps (> ½ line height = new paragraph); match anchors by substring. `pipeline/ingest.py:page_paragraphs`.

### Print header/footer leaked into the transcript
- **Symptom:** the URL and the title line repeated every ~12–15 paragraphs inside the body.
- **Cause:** the footer URL wraps over two lines, so a full-line URL regex missed it; the title shares a block with the timestamp.
- **Caught by:** counting URL/"SPECIAL…YEAR" matches per body.
- **Fix:** drop blocks by **position** (outer 3.5% of the page) instead of by regex.

### Site synopsis left at the top of a set: four different layouts
- **Symptom:** sets opened with the site's description, release info, critics' quotes.
- **Cause:** the site's post layout changed over the years:
  1. Newer posts: "Transcript" heading + study disclaimer (ISMO, Arcuri, Hamilton, Morril).
  2. Older posts: synopsis ending in a `* * *` divider (Gulman, Morgan, Bargatze, Notaro, Birbiglia).
  3. Oldest: infobox + title + blurb only.
- **Caught by:** user eyeballing `body.txt` files.
- **Fix:** rules in priority order (heading → divider → keyword fallback), then made **explicit
  per-set counts** in `data/body_overrides.yaml` the source of truth; heuristics only *propose*
  them via `ingest --suggest`. Output verified byte-identical across the refactor.

### `* * *` divider: two traps
- **Symptom 1:** the divider rule never fired.
- **Cause 1:** "is there a stage marker above the divider?" matched the post title's "(2023)".
- **Fix 1:** only *reaction* markers (laugh/applause/cheer) count.
- **Symptom 2:** Birbiglia's synopsis survived.
- **Cause 2:** his synopsis + critics' quotes ran to paragraph 16; search window was 15.
- **Fix 2:** window widened to 25; Birbiglia trim set to 17 by hand.
- **Also:** a divider *with* reactions above it is a section break inside the set and must be kept.

### Disclaimer tail survived as the "first line"
- **Symptom:** ISMO, Arcuri and Morril opened with a sentence fragment about rights belonging to the creators.
- **Cause:** the study disclaimer wraps across a page break → two paragraphs; only the first was skipped.
- **Caught by:** user, then an automated "body starts lowercase" check across all sets.
- **Fix:** +1 to those trims; `--suggest` now skips lowercase continuations after the disclaimer.

### "MORE … Stand-Up Transcripts" footer kept at the end
- **Cause:** two word orders ("MORE <name> Stand-Up Transcripts" and "MORE Stand-Up Transcripts <name>").
- **Fix:** looser regex; footer handled via `skip_trailing`.

### Lowercase-start check gave false positives
- **Cause:** bodies start with markers like `[audience cheering]`, whose text is lowercase.
- **Fix:** strip leading markers before testing. Lesson: test the check, not just the data.

---

## Data pipeline: beats and lines (S3)

### Beat length predicted the label
- **Symptom:** 77% of beats labeled "laugh"; median 23 words for laugh beats vs 74 for no-laugh.
- **Cause:** beats are cut *at* laughs, so negatives only come from chunking long quiet stretches.
  A model would learn "short = funny".
- **Caught by:** summary stats printed after the first S3 run.
- **Fix:** line-level labels (`lines.jsonl`): punchline line gets the level, setup lines 0, each
  with 150 words of preceding context. Laugh rate 0.15; median 7 vs 6 words.

### Four failing tests were test bugs, not code bugs
- Reactions placed 1 word apart merged (by design, < 3 words); a 2-word "punchline" was folded
  into the setup (by design, < 3 words); a context assertion was wrong; the debug view had a
  double space before tags (cosmetic, fixed). Lesson: when a new test fails, decide whether
  the test or the code is wrong before changing either.

---

## Data pipeline: tuning S4 without overfitting

### First evaluation: Gemini under-segments and finds no callbacks
- **Result (Bargatze, tuning set):** bits F1 0.75 (P 0.89, R 0.65), threads F1 0.61, callbacks 0/3;
  even-spacing baseline F1 ~0.21. Half the missed boundaries were same-topic splits, half new topics.
- **Changes (general, not set-specific):** prompt gives an explicit procedure (topic map, standalone
  test, `continues` only for same subject, callback check on every bit incl. thematic ones);
  local callback candidates (corpus-rare terms that vanish ≥ 30 lines and return) fed as hints.
- **Hint pass alone:** finds 1/3 Bargatze callbacks (the lexical one); the 2 thematic ones need the
  LLM. A first version found 0/3: 60-line gap too strict, and coverage was checked too narrowly
  (the return landed in the setup bit just before the callback bit).
- **Protocol:** Bargatze = tuning set; Birbiglia = held-out test, labeled and scored once.
- **Second run (new prompt + hints), 2026-10-21:**

  | Bargatze | first run | new prompt |
  |---|---|---|
  | bits (ref 52) | 38 bits, F1 0.75, P 0.89, R 0.65, Pk 0.172 | 48 bits, **F1 0.80**, P 0.83, **R 0.77**, **Pk 0.146** |
  | threads (ref 37) | 21, F1 0.61, Pk 0.277 | 22, F1 0.63, Pk 0.263 |
  | callbacks (ref 3) | 0 found | **0 found** (even the hinted front-door one) |

  Granularity now close to human (median 17 lines, same as labels). Still over-uses `continues`
  (22 vs 37 threads). Callbacks unchanged: asking for them inside the segmentation call doesn't
  work. The call took 1145 s with two network retries.
- **Next idea (general):** a separate, smaller callback pass over the bit list (premises + hooks +
  lexical candidates) instead of the full transcript, so the model focuses on that one judgement.

### Gemini free-tier quota exhausted mid-run
- **Symptom:** repeated HTTP 429 with escalating backoff; probe says "exceeded your current quota".
- **Cause:** daily request quota on the free tier; long thinking-model calls (300–530 s) under load.
- **Fix:** stopped the run (outputs from the old prompt are re-requested automatically via the
  prompt hash). Options: wait for the daily reset, paid tier, or another provider.

---

## Hand labeling

### Label template had lines split in two
- **Symptom:** 18 fragments without a `[NNNN]` number in the Bargatze work file; import would fail.
- **Cause:** folding a short final line ("So…") into the previous one can cross a paragraph
  break, so the line text kept a newline.
- **Caught by:** structural check while reviewing the user's labels.
- **Fix:** line text is whitespace-normalized in `pipeline/beats.py`; the user's file was repaired
  by rejoining fragments (all marks kept, backup taken). Line indices unchanged.

---

## Data pipeline: bits (S4–S5, Gemini)

### First LLM pass invented a callback and called the goodnight the closer
- **Symptom:** a bit "called back" to the bit right before it; the thank-you segue was the closer.
- **Cause:** prompt allowed any earlier bit; position was assigned by raw order.
- **Caught by:** reading the paraphrased premise list for Maniscalco.
- **Fix:** validator requires callbacks ≥ 2 bits back (errors fed back to the model, which
  self-corrected on attempt 2); opener/closer skip segues. Cached outputs are re-validated on
  load so tightened rules can't be bypassed by an old cache.

---

## Data pipeline: markers and labels (S1–S2)

### Marker offsets drifted later and later through each set (**real bug**)
- **Symptom:** text before each laugh ran into the next sentence by 1, 2, 3… characters.
- **Cause:** removing a marker leaves a double space; spaces were collapsed *after* offsets were computed, so each removed marker shifted all later offsets by one.
- **Caught by:** user running `scripts/inspect_events.py laughs` on Bargatze.
- **Fix:** collapse whitespace per segment while building the clean text; verification that compares text before every marker → **0 mismatches across ~3,500 markers**. Lesson: build an automated alignment check, don't trust offsets by construction.

### Inconsistent reaction vocabulary (~425 distinct bracket strings)
- **Cause:** every transcriber writes reactions differently ("laughter", "audience laughing", "crowd laughs").
- **Fix:** versioned lexicon `data/markers.yaml`; unknown *reaction* strings stop the run; `--draft` proposes entries for review. Non-reaction directions are bucketed by keyword instead of listed (would be ~300 entries).

### Performer sounds mistaken for audience reactions
- **Symptom:** "jeff laughs", "tom chuckles" proposed as crowd-work people.
- **Cause:** subtitle convention: unattributed or named-performer sounds are the *speaker*. The names were the comedians' own first names.
- **Fix:** mapped to `performer` in the lexicon; documented in the wiki.

### "laughing continues" attributed to the performer
- **Cause:** draft heuristic saw no "audience" word.
- **Effect:** Leanne Morgan's big-laugh count was 1 instead of 33.
- **Fix:** any "… continues" marker extends the preceding audience reaction.

### Half the transcripts barely tag laughs
- **Symptom:** Bamford, Notaro, Papa, Martin, Hamilton showed ~0 audience laughs.
- **Cause:** sparse transcribers (480+ words per tagged laugh vs 26–80 for dense ones), not misattribution.
- **Fix:** density gate (`laugh_labeled` only if ≤ 150 words/laugh). Only **11/20** sets carry laugh labels; Bamford had to be swapped out of the held-out set for Birbiglia. **8 training sets** remain: a data bottleneck.

### Applause counted as a laugh
- **Symptom:** a cheer for the comic's hometown labeled as an applause-break laugh.
- **Caught by:** user's alignment check.
- **Fix:** applause/cheers alone → level 0 + `applause` flag; level 4 requires a laugh. Level-4 counts dropped sharply (e.g. Notaro 23 → 0).

### Songs treated as spoken material
- **Symptom:** walk-on music lyrics (♪ lines) at the start/end of sets; Bamford sings mid-set.
- **Cause:** lyric lines aren't bracketed and often lack a closing ♪.
- **Fix:** paragraph-level song handling before marker extraction: edge songs dropped, mid-set singing → one `[sings]` event, speech in mixed paragraphs kept. No lyrics stored.
- **Note:** `body.txt` still shows songs by design (raw cut); `clean.txt` doesn't.

### "normal voice" isn't an act-out
- **Cause:** it marks the *end* of a character voice (Bamford 37×, Martin 8×).
- **Fix:** new category `act_out_end` so character spans can be measured.

---

## rJokes prep (R1)

| Problem | Cause | Fix |
|---|---|---|
| Split files have no dates | Only `preprocessed.csv` (raw upvotes) has timestamps | Join on whitespace-normalized text: 99.6% of rows match; 1,081 jokes left without a year |
| Same text, very different scores | Reposts: 6,971 texts posted more than once, median spread 2 on the ln scale (~7x upvotes) | Dedupe keeps the max score + `n_posts` + `score_spread`. The spread is the **noise ceiling** for any joke scorer: don't expect high correlation |
| Repost in train and test | Splits were random over posts, not texts | The first split in train→dev→test order owns a text |
| r/Jokes grew ~1000x 2008→2019 | Raw scores aren't comparable across years | `pct` = score percentile within the post's year; use `pct`, not `score`, for priors |
| `pytest` missing | System Python has no pytest | Run tests with `.venv/Scripts/python -m pytest tests -q` |
| Hand-check CSV → `UnicodeDecodeError: 0x92` (2026-10-31) | Excel "CSV" saves in cp1252: smart quotes are bytes 0x92–0x94, not UTF-8 | Read once with `encoding="cp1252"`, write the tracked copy (`data/labels/rjokes_hand_check.csv`) as UTF-8. Save as **"CSV UTF-8"** in Excel next time |
| `test_docs` → `UnicodeDecodeError: 0xb1` in report.md (2026-11-01) | A `sed` replacement typed through the agent's shell wrote "±" as one Latin-1 byte, not UTF-8 | Caught by the docs test; byte repaired. Same lesson as the backslash rows: **edit text with non-ASCII characters through the Edit tool, not `sed`** |
| Labeling job "OOM at 76k" after #23 merged | Job started before the cluster checkout was pulled; Python had already loaded the old script | `sacct` start time vs `git reflog` pull time. **Pull, then submit**; check `git log -1` in the job log |
| Hand-check CSV has joke text but `data/processed/` is gitignored | Evidence for the notebook would be lost | Tracked copy keeps ids, judge labels, human verdict/mode/flags only; free-text safety notes mapped to judge flag names (`other:<topic>` when outside the schema) |

Result: 432,457 rows → 402,981 after dedupe + 4–200 word filter (train 323,573 / dev 39,764 /
test 39,644). 75% carry no topic flag; scores are heavy at 0 (35%).

## Data sourcing and legal

### Scraping the transcript site
- **Issue:** robots.txt allows crawling, but the site's Terms of Use prohibit saving or reformatting material by electronic means; transcripts are fan transcriptions of copyrighted specials.
- **Decision:** no bulk scraper. Transcripts were downloaded manually for local study; corpus stays gitignored; never used as imitation text; memorization guard planned.

### Reference papers in a public repo
- **Issue:** one paper (HumorGen) is an anonymous submission under review.
- **Fix:** `references/*.pdf` and converted markdown gitignored; README index tracked.

---

## Explorer cluster setup (2026-10-31)

Getting one 50-joke labeling smoke test to run took most of a day. In order:

| Problem | Cause | Fix |
|---|---|---|
| OOD "JupyterLab ML Courses": "does not supply a sub app form file" | Server-side OOD app config | Shell Access + `srun`/`sbatch`; no JupyterLab needed |
| `pip install vllm` "stuck", then `^C` → no vllm | The final "Installing collected packages" step writes GBs to network storage silently for 10–20 min | Use `uv` (visible progress, faster); never interrupt installs; run them as batch jobs |
| `import vllm` OK but `from vllm import LLM` → `TypeError: unsupported operand ... 'type' and 'NoneType'` | Default `python3` is **3.9**; vLLM uses 3.10+ syntax, and `import vllm` is lazy so the plain import hid it | Check the *real* import; build the venv on `module load python/3.13.5` |
| `uv venv --python 3.12` failed after 23 min | uv's Python download from GitHub timed out (3 retries) | Use the cluster's 3.13 module as the interpreter |
| `python3 -m pip` → "No module named pip" on compute nodes | System 3.9 ships without pip | Install `uv` with the 3.13 module's pip |
| `python3 -m pipeline.rjokes` → `Killed` | Ran on the **login node**, which caps memory | Heavy work only on `cNNNN`/`dNNNN` compute nodes |
| `$C` empty → "failed to create directory /venv" | New shell; exports don't carry across sessions | Full paths in instructions |
| First GPU job landed on a **P100** | `--gres=gpu:1` takes any card; vLLM doesn't support the P100 | Request `--gres=gpu:v100-sxm2:1` |
| vLLM 0.32 on V100: "NVIDIA driver on your system is too old (found version 12030)" | Driver supports CUDA ≤ 12.3; vLLM's torch was built for newer CUDA (which also drops Volta) | Drop vLLM; `transformers` + `torch==2.5.1+cu121` |
| `Could not import module 'AutoModelForCausalLM'` | transformers 5.19 needs a newer torch than 2.5 | Pin `transformers>=4.56,<5` |
| `operator torchvision::nms does not exist` | torchvision left over from the vLLM install, built for the newer torch | Uninstall torchvision/torchaudio; pinned `requirements-cluster.txt` |
| Install sessions killed (`TIMEOUT` at 30 min; 17 GB peak on a 16 GB request) | Installs ran inside interactive GPU/CPU sessions | Installs as `sbatch --wrap` jobs with 32 GB / 1 h on `courses` |
| `sbatch` printed nothing | A blank line after a trailing backslash → `sbatch` waited for a script on stdin | Paste commands as one line |
| GPG signing timed out on commit | Passphrase cache (10 min) expired; the pinentry window popped up unseen | Watch for the passphrase window when committing |

Result: V100-SXM2-32GB, float16, Qwen2.5-7B-Instruct, **3.4 jokes/s, 0/50 parse failures**.
The working combination is pinned in `explorer/requirements-cluster.txt`.

Later, during the 90k extension run:

| Problem | Cause | Fix |
|---|---|---|
| `torch.OutOfMemoryError` at 72k/90k (tried 3.1 GB, 2.6 GB free, 4.5 GB reserved but unused) | Batches are length-sorted, so a chunk's 16 longest jokes share one batch; a few very long posts overflow 32 GB. Fragmentation made it worse | `split_on_oom`: halve the batch and retry; one joke that still doesn't fit is logged as a parse failure. `expandable_segments:True` in the sbatch. Saved chunks were safe; resubmitting resumes |
| An `--n` extension came close to the 4 h limit (60k new jokes ≈ 3.3 h) | `--time` was sized for the first 30k | `--time=08:00:00` (courses-gpu maximum) |

---

## Tooling and environment

| Problem | Cause | Fix |
|---|---|---|
| Python 3.13 venv creation failed | 3.13 registered with `py` launcher but its exe was missing | Use 3.14; revisit for audio/ML wheels in week 5 |
| `pymupdf4llm` 2.x installed nothing | Empty wheel on Python 3.14 | Pinned 0.3.4 |
| `create-next-app` installed Next 15.5.4 | Version had CVE-2025-66478 | Upgraded to patched 15.5.x |
| `python -I -m pipeline.ingest` → module not found | `-I` removes the repo root from `sys.path` | Run as plain `python -m` (our own code; PDFs are only read as data) |
| `sed` "unknown command" | MinGW sed on PATH | Use the Edit tool / Python for file edits |
| `UnicodeEncodeError` printing ♪ | Windows console uses cp1252 | `PYTHONIOENCODING=utf-8` |
| `venv/Scripts/python` not found | Folder is `.venv` (leading dot) | Use `.venv/...` or plain `python` when activated |
| Gemini `gemini-2.5-flash` → 404 | Listed by `/models` but retired for generation | Use `-latest` aliases; check `/models` |
| Gemini Flash TTFT 18–38 s | Thinking model; `reasoning_effort: none` didn't help on free tier | Live loop uses `gemini-flash-lite-latest` (~0.6 s); Flash only offline |
| Merged PR #4 never reached `main` | #4 was stacked on `feat/sim-env` and merged there after #1 had already merged that branch into `main`; same for a commit pushed to `docs/wiki` after #2 merged | Landing PRs #5, #6. Rule: **no stacked PRs**; branch from `main` and wait for the base to merge, or retarget the child to `main` before merging the parent |
| `python -m sim.evaluate_policies` → SyntaxError on main | A Python one-off patch wrote a real newline inside a string literal (the backslash-n escape was lost through a heredoc); no test imported the module, so CI-by-pytest stayed green | Fixed; `tests/test_scripts_import.py` now imports every module in sim/, pipeline/, scripts/, explorer/. Prefer the Edit tool over heredoc+Python for code patches |
| Same escaping bug, three more times (2026-10-28 → 31): `evaluate_shift.py`, the sbatch file (a literal backslash-n broke the job's command line), and the guard test itself | **Root cause:** text sent through the agent's shell-heredoc path has `\\` collapsed to `\`, so any backslash escape written that way changes meaning. Not a Python or bash quirk | Caught by the import smoke test, then by `test_shell_scripts_have_no_literal_backslash_n`. **Rule: never write code or scripts containing backslashes via shell heredocs; use the Edit/Write tools** |
| Mermaid diagrams blank in preview | Older third-party preview extension | Official `mermaidchart.vscode-mermaid-chart`; recommended in `.vscode/extensions.json`. All diagrams also validated with the Mermaid parser |
| `pip install kokoro` fails on Python 3.14: `blis` won't compile (2026-11-01) | Kokoro needs spaCy, whose build dependencies don't support 3.14; `py -0p` also listed a 3.13 that was no longer installed | Backend venv on Python 3.12 from `uv python install 3.12`; recorded in `backend/requirements.txt` and its README |
| C1b's choice quality falls through the show even with no turn (2026-11-01) | `LinQRefit.update` grew `A` after each bit but `b` only at the show's end, so values dropped mid-show with no new evidence (caught by review) | Fixed at the user's request: `A` and `b` change together at the show's end, and the refit uses values from before the show was added. Tests cover both. Lesson: test that a ridge model's `A` and `b` change together |
| #28–#30 show as merged but `main` doesn't have them (2026-11-01) | Each stacked PR merged into its base branch, and none was retargeted to `main` as its description said it should be. Caught when `origin/main` stopped at #27 | Landed the stack through #32. Retarget a stacked PR to `main` before merging it, then confirm with `git merge-base --is-ancestor` |
| Safety filter misses spaced-out insults like `a m o r o n` and `gentleman's tie` (2026-11-08) | The spaced-letter join regex matched the article "a" as part of the run (`amoron`), and the aimed-at-someone marker set compared whole tokens, so `gentleman's` never matched `gentleman` (caught by the held-out lines in Exp 19) | Not fixed in T8: the bar was missed and a held-out miss is reported, not re-tuned. Fix both in the follow-up and re-score on fresh held-out lines |
| Real-bank false rejects came from capitals after quotes (2026-11-08) | The name rule read `'minor fixes.' Nobody` and `'Next Steps'` as names: it split sentences on the period but not on a closing quote, and read quoted capitals as names (12 of 14 bank rejections; caught by the review on the merged bank) | Quoted spans are lowercased before the name check, sentences end after a closing quote, possessives match their base word and the spaced-letter join drops a lone article. Test with the bank's own lines |
| LLM check failed 29% of attempts with 429 and 503 on the free tier (2026-11-08) | Gemini flash-lite refused or dropped 10 of 35 attempts in Exp 19, and one run lost a whole bank batch to a 503, which read as 15 false rejects (caught by the eval's failure count) | Eval retries 429 and 503 with backoff and reports the retry count. The live filter must not hide this: a failed call rejects the bit's lines, so T9 needs the bank as the fallback |
| Backend tests could not run: `ModuleNotFoundError: numpy` (2026-11-08) | The plan said to use `backend/.venv`, a stale Python 3.14 venv with no numpy or kokoro. Kokoro needs Python 3.12 (`backend/README.md`). Caught by the T2 builder and review | Run backend tests with a Python 3.12 venv that has `requirements.txt` and `requirements-dev.txt`. The plan's test line now says so. Recreate the stale venv with 3.12 |
| Exp 20's first run showed within-bit gaps of exactly 0.30, 1.00, 1.50 and 2.50 s, the hold lengths (2026-11-14) | `silences()` counted a hold only if the `gap` row's timestamp was earlier than the next play's start. On Windows `time.time()` is coarse, so the row and the next play often carry the same instant, the hold was dropped and its whole length read as dead air. Caught by review, which saw the gaps equal the hold table and reproduced it with one fake show | The hold's end is found by log order (the `gap` row after each `joke_played` row) and combined with the player's audio end by `max`, never filtered by comparing two clocks. Tests: a gap row equal to the next start gives 0, a real gap is kept |
| No usable backend venv for T9's tests and Exp 20 (2026-11-14) | The main checkout's `backend/.venv` is still the stale Python 3.14 one (no numpy, no kokoro; see the 2026-11-08 row above). The builder made a throwaway 3.12 venv outside the repo before the coordinator pointed to the demo checkout's 3.12 venv | Backend tests and Exp 20 ran on the demo checkout's venv. The plan's "use `backend/.venv`" line is still wrong until the main venv is rebuilt on 3.12 |
| Exp 20's third run: 80 jokes fell back to the bank (2026-11-14) | After the thread decision the prompt asked for candidates "slot by slot" and the model answered thread-major (3 threads of 5 lines), while the parser expected joke-major (5 jokes of 3 candidates), so every reply failed with `expected 5 jokes of 3 candidates, got [5, 5, 5]`. Caught by the reviewer reading the run's log, not by a test: the tests built replies in the shape the parser wanted | One explicit thread-major shape in both prompt and parser, any other shape rejected (tests: a real-shaped reply parses, the old joke-major shape and string rows are rejected); three single-bit probes against the real model before the rerun |
| A retrying call could run well past its deadline (2026-11-14) | `Client.chat` checked `deadline_s` only before the backoff sleep, and every attempt still got the full `timeout`, so a 10 s call with a 12 s deadline could take over 20 s. Caught by review | Each attempt's timeout is clipped to the time left, and no attempt starts with under 0.5 s left (test: a server that hangs, total wall time within the deadline plus slack). httpx's timeouts are per phase and its read timeout is per chunk, so the attempt also reads the body in chunks and stops at its timeout; one stalled chunk is still bounded only by the read timeout, so the total is not strictly guaranteed within the deadline |
| One line the voice could not say would have aborted a show (2026-11-14) | `_prepare` awaited the first line's render and `_joke` the others, so Kokoro's "no audio produced" (the model can write emoji or symbols only) propagated and ended the show. Caught by review | A line that fails to render is skipped with the reason in its item, then the next passing candidate, then the bank, and only then the joke; `parse` rejects lines with no letters. The old crash test now crashes the player, since a voice failure no longer aborts |
| Two concurrent shows could still synthesize at once (2026-11-14) | The lock that serialized Kokoro was created per show, but the engine is one process-wide object (`main.get_tts`). Caught by review | The lock is `app.tts.SerialTTS`, wrapped once around the engine (test: two shows sharing one engine, peak concurrency 1, and 2 or more without the wrapper) |
| Dead-air numbers could pair a repeated line with the wrong hold (2026-11-14) | `silences()` keyed plays and holds by line text, so a bank line used twice took another play's hold. Caught by review | Dead air is read from the log in row order, now that bridge lines are `bridge` rows (test with a repeated line) |
| `run_notebook.py` failed on Exp 19 with `KeyError: LLM_BASE_URL` on a fresh checkout of main (2026-11-14) | `evaluate_safety.digest()` hashed the fixtures' raw bytes, and line endings depend on the checkout: git's `autocrlf` writes CRLF, while files written by agent tools are LF. The cache was keyed in a worktree with LF test lists and a CRLF bank, so every other checkout missed the cache and tried a live API rerun. Caught by the T9 builder; cause found by review (exactly one of the eight line-ending combinations matches the cached key) | The digest normalizes CRLF to LF. The cached `fixtures` key was re-keyed to the normalized digest without a rerun, since the content is byte-identical once line endings are normalized |
| The recording held 65% of the audio, so every timestamp in the WAV drifted early (2026-11-15) | The WAV grew at 10,444 samples per wall second instead of 16,000, and 5-9% of samples were exact zeros; earlier shows lost 5%, 15% and 26% (one on Bluetooth). The capture context was created with `sampleRate: 16000` over a 48 kHz microphone stream, and Chrome's resampling of a non-native context rate underran; the worklet also returned early on a quantum with no input, so a missing quantum did not advance time. Caught by the user's Chrome test, then the reviewer's measurement of samples against wall clock; the synthetic tests wrote frames straight into `Recorder` and never saw the browser's rate | **Round 2's fix (native-rate context, downsampling in the worklet) did not help:** the user's Chrome test with Kokoro on the CPU still recorded 65-68% (device 48 kHz, 0 silent quanta), so the loss is the render thread missing deadlines, not resampling. Torch thread caps and priority did not fix it either: 8 threads 65-68%, 4 threads 74%, 2 threads 99.8% (but Kokoro then runs at 1.0x real time, too slow for render-ahead, against 0.35x at 8), 8 threads with the backend at BelowNormal 74%. **Final fix:** read the mic track's frames directly (`MediaStreamTrackProcessor`) on the capture clock, fill gaps with silence from the frame timestamps and trim overlaps, so the WAV timeline no longer depends on any render thread; the 16 kHz downsampler (windowed-sinc, phase kept) stays; `audio_health` is computed from capture timestamps; `audio_start` gives the server the capture time of sample 0. `npm run test:recorder` simulates frames with a 200 ms gap and a 5 ms overlap and checks every sample against absolute time; a headless Chrome with a fake mic and a busy main thread produced exactly real time. Earlier mitigations that stayed: the browser sends `audio_health` every ~5 s and the server warns below 99%, also on close (`Recorder.coverage`). `npm run test:recorder` checks 16,000 samples per second of input from 48 kHz, 44.1 kHz, 32 kHz and 16 kHz, the tone frequency, alias rejection and chunking-independence. **Still to be measured on the user's laptop with Kokoro running** |
| `winsound.Beep` made no sound, so Exp 21's first run had nothing to record (2026-11-15) | `winsound.Beep` uses a legacy beep path that many laptop drivers ignore; it returned without error. Caught by the user hearing nothing during the run | `play` plays an in-memory 2 kHz, 100 ms WAV through `winsound.PlaySound(SND_MEMORY)`, the output device the robot's voice uses, and saves `clicks.json` after every beep |
| The recording had no beeps and the robot's voice was faint (2026-11-15) | The microphone's Windows "Audio enhancements" ran the driver's echo cancellation, which removes what the laptop itself plays; it sits below the browser, so `echoCancellation: false` did not reach it. Caught by the user's first Exp 21 run (no beeps in the WAV) and the reviewer's matched-filter search | Enhancements off for the mic (and vendor noise-cancellation apps quit), written into design.md's setup list; the second run recorded all beeps at 23x to 31x the noise |
| `check` reported "0 clicks matched" on a recording that held 30 beeps (2026-11-15) | `detect_clicks` had an absolute threshold floor of 0.01, above the beeps' recorded amplitude (about 0.003), and `MATCH_S` was 0.25 s while every beep landed 224 to 262 ms late (a constant output latency), so even detected beeps did not match. Caught by the reviewer's own matched-filter measurement, which found 30/30 | A matched filter for the 100 ms 2 kHz tone with a threshold relative to the recording's noise only, `MATCH_S` 0.5 s, the strongest detection in the window wins, and beeps outside the recording are counted apart (tests: beeps at 0.003 amplitude under louder broadband noise are all found; a constant 240 ms offset with 20 ms jitter passes the spread bar) |
| The safety check timed out on 5 of 8 bits and one show was 75% bank (Exp 22, 2026-11-15) | The filter's 6 s attempt / 8 s deadline was sized for a 1.3 to 2 s call; the provider sometimes stalls past it, and a failed call rejects every line of the bit (fail closed). One malformed verdict reply rejected a bit the same way. The brief assumed the wait was hidden by the previous bit's playback; it is not (`run_show` prepares a bit after the previous one ends), only bridge lines cover it | The acceptance run's bank-share count, read from the candidates' reject reasons (73 ReadTimeouts, 15 JSONDecodeErrors) | Filter 12 s / 25 s with 3 retries; one fresh re-ask on a bad or partial reply; `MAX_BRIDGES` 3 to 5 to cover the longer wait; tests on a fake clock |
| One generator call took 361.7 s in the Exp 20 rerun, far past its 12 s deadline, and ended in `RemoteProtocolError` (2026-11-15) | Not found. `llm.Client` bounds each attempt by `timeout` and a total by `deadline_s`, and the same show's dead air stayed at 0.007 s, so the wall clock and the show's clock stalled together: most likely the laptop slept or the run was suspended, not a provider stall. The call is in the run's `generator.seconds`, unexplained | Reading the per-call seconds of the rerun | Nothing changed. If a long call reappears without a suspend, check the connect and write phases of the httpx timeout and the read of the first byte |

---

## Process lessons

- **Eyeball checks found every real bug.** Counts looked plausible each time; reading 10 aligned lines exposed the offset drift. Keep `scripts/inspect_events.py` in the loop for every stage.
- **Fail loud > guess.** Lexicon and trim files stop the run on anything unseen; a human approves once.
- **Determinism checks** (`md5sum` of outputs) made refactors safe.
- **Don't print transcript text in tooling output**: masked shapes and counts were enough to debug most issues.
