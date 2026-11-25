# Decisions

What we decided and why, one line each. Click an ID for the full record (options, numbers, evidence).

| ID | Date | Decision |
| --- | --- | --- |
| [D-01](agent/decisions-detail.md#d-01) | Oct 20 | Audio only, no camera: faster, and the robot can't joke about how people look. |
| [D-02](agent/decisions-detail.md#d-02) | Oct 20 | Learn from laughs per joke; report the whole-show rating. Ratings are too rare to learn from, and raw laugh counts can be gamed. |
| [D-03](agent/decisions-detail.md#d-03) | Oct 20 | One robot personality; a bandit picks which comedy style to use. Switching whole personas felt incoherent. |
| [D-04](agent/decisions-detail.md#d-04) | Oct 20 | Start with an AI judge; replace it later with a scorer trained on our own audience votes. |
| [D-05](agent/decisions-detail.md#d-05) | Oct 20 | Main data is our own show logs; rJokes and published methods get us started. |
| [D-06](agent/decisions-detail.md#d-06) | Oct 20 | Research questions: does adapting beat a fixed style? Does a trained scorer beat the AI judge? |
| [D-07](agent/decisions-detail.md#d-07) | Oct 20 | First milestone is the Dec 15 course demo, with the RL part proven in a simulator. Date later moved (D-44). |
| [D-08](agent/decisions-detail.md#d-08) | Oct 20 | Before a show the operator sets venue, energy and age group. Audio can't tell age. |
| [D-09](agent/decisions-detail.md#d-09) | Oct 20 | Teen audiences always get a stricter safety filter. |
| [D-10](agent/decisions-detail.md#d-10) | Oct 20 | The system writes the next bit based on how the room is reacting. |
| [D-11](agent/decisions-detail.md#d-11) | Oct 20 | Comedy transcripts teach structure and laugh timing only; we never copy their jokes. |
| [D-12](agent/decisions-detail.md#d-12) | Oct 20 | Transcript laughs are scored 0 to 4; "laughed or not" is the main label. |
| [D-13](agent/decisions-detail.md#d-13) | Oct 20 | Opener and closer are positions in the set, not kinds of bits. |
| [D-14](agent/decisions-detail.md#d-14) | Oct 20 | Only transcripts that mark laughs often are used for laugh labels. |
| [D-15](agent/decisions-detail.md#d-15) | Oct 21 | Applause without laughter doesn't count as a laugh. |
| [D-16](agent/decisions-detail.md#d-16) | Oct 21 | Transcripts split into two datasets: with laugh labels, and structure only. |
| [D-17](agent/decisions-detail.md#d-17) | Oct 21 | Callbacks are something the planner can choose. Later deferred (D-46). |
| [D-18](agent/decisions-detail.md#d-18) | Oct 21 | The laugh predictor learns from single lines, because passage length alone gave the answer away. |
| [D-19](agent/decisions-detail.md#d-19) | Oct 21 | Use free Gemini to label transcripts, sending as little text as possible. Replaced by D-20. |
| [D-20](agent/decisions-detail.md#d-20) | Oct 21 | Send each whole set to Gemini in one call; it answers with line numbers, never quotes. |
| [D-21](agent/decisions-detail.md#d-21) | Oct 21 | A bit is the smallest piece that works on its own; bits on one topic form a thread. |
| [D-22](agent/decisions-detail.md#d-22) | Oct 26 | Pause transcript work and build on rJokes and a simulator first: behind schedule, and less legal risk. |
| [D-23](agent/decisions-detail.md#d-23) | Oct 26 | Plan a long set bit by bit, not as one long text, so the show can change live. |
| [D-24](agent/decisions-detail.md#d-24) | Oct 26 | University GPU cluster for heavy batch jobs; everything else runs on the laptop. |
| [D-25](agent/decisions-detail.md#d-25) | Oct 27 | The director picks a style once per bit, not per joke: comedians commit to an angle, and 5 jokes give a steadier signal. |
| [D-26](agent/decisions-detail.md#d-26) | Oct 27 | The director uses LinUCB: it lost the fewest laughs in simulation. |
| [D-27](agent/decisions-detail.md#d-27) | Oct 28 | Adding "how the last bit did" as an input didn't help; keep the inputs as they are. |
| [D-28](agent/decisions-detail.md#d-28) | Oct 28 | Keep 8-bit (10-minute) shows; longer shows didn't help the director notice a mood change. |
| [D-29](agent/decisions-detail.md#d-29) | Oct 28 | Remembering tonight's results (step A) failed its test; not adopted. |
| [D-30](agent/decisions-detail.md#d-30) | Oct 31 | A per-show correction (step B) failed too: it kept repeating what worked and the room got tired. |
| [D-31](agent/decisions-detail.md#d-31) | Oct 31 | The simulated audience's style preferences come from the HumorGen paper's judge scores. |
| [D-32](agent/decisions-detail.md#d-32) | Oct 31 | rJokes is used for style statistics only, never as jokes; its safety labels proved unreliable. |
| [D-33](agent/decisions-detail.md#d-33) | Oct 31 | Docs may quote a few rJokes examples, never the dataset. |
| [D-34](agent/decisions-detail.md#d-34) | Nov 1 | Plan for comparing human (upvote) and AI judge rankings of styles, with pass criteria set before running. |
| [D-35](agent/decisions-detail.md#d-35) | Nov 1 | The human ranking compares jokes of similar length (10 length groups). |
| [D-36](agent/decisions-detail.md#d-36) | Nov 1 | Humans and the AI judge agree only weakly, so the simulator is tested with both tastes. |
| [D-37](agent/decisions-detail.md#d-37) | Nov 1 | Longer jokes get slightly more upvotes, so styles are compared only among similar-length jokes. |
| [D-38](agent/decisions-detail.md#d-38) | Nov 1 | Plan for checking whether other joke features skew the human ranking. |
| [D-39](agent/decisions-detail.md#d-39) | Nov 1 | No other feature mattered; the human ranking stands. |
| [D-40](agent/decisions-detail.md#d-40) | Nov 1 | That check's rule had a gap; an extra test confirmed the main findings. |
| [D-41](agent/decisions-detail.md#d-41) | Nov 1 | Pass criteria for testing the director under human taste. |
| [D-42](agent/decisions-detail.md#d-42) | Nov 1 | LinUCB still wins under human taste; keep it. |
| [D-43](agent/decisions-detail.md#d-43) | Nov 1 | Voice is Kokoro, run locally on the CPU: free, and no network needed at the venue. |
| [D-44](agent/decisions-detail.md#d-44) | Nov 1 | Demo moves to Dec 8; presentation and report are due Dec 14. |
| [D-45](agent/decisions-detail.md#d-45) | Nov 1 | Banter and roast replies are cut from the demo to make time. |
| [D-46](agent/decisions-detail.md#d-46) | Nov 1 | Try show-long planning now (step C1); callbacks come after the demo. |
| [D-47](agent/decisions-detail.md#d-47) | Nov 1 | Step C1 uses linear Q-learning, gets the same test as steps A and B, and runs once. |
| [D-48](agent/decisions-detail.md#d-48) | Nov 1 | C1's first run failed, likely because of a scaling mistake in its design. |
| [D-49](agent/decisions-detail.md#d-49) | Nov 1 | With the scaling fixed (C1b) it still failed; the demo keeps LinUCB. A code bug was found afterwards. |
| [D-50](agent/decisions-detail.md#d-50) | Nov 1 | With the bug fixed it still fails; LinUCB stays. |
| [D-51](agent/decisions-detail.md#d-51) | Nov 1 | rJokes may be used to test AI judges, still never as jokes. |
| [D-52](agent/decisions-detail.md#d-52) | Nov 1 | Train our own joke model (SFT, then DPO) for the demo, from premises we write. |
| [D-53](agent/decisions-detail.md#d-53) | Nov 1 | Report the true best possible score beside the "best next bit" benchmark; they differ by under 1 laugh a show. |
| [D-54](agent/decisions-detail.md#d-54) | Nov 1 | Every experiment gets a chart, or a reason it doesn't need one. |
| [D-55](agent/decisions-detail.md#d-55) | Nov 8 | The live show is one program that writes one log file per show. |
| [D-56](agent/decisions-detail.md#d-56) | Nov 8 | At each show start, the director relearns from the saved logs. |
| [D-57](agent/decisions-detail.md#d-57) | Nov 8 | The operator rates each joke's laugh with a key press (0 to 3); that is the reward. |
| [D-58](agent/decisions-detail.md#d-58) | Nov 8 | The director tries a random style 20% of the time, so the logs can test other directors later. |
| [D-59](agent/decisions-detail.md#d-59) | Nov 8 | Record show audio, only with the room's consent, to train a laugh detector later. |
| [D-60](agent/decisions-detail.md#d-60) | Nov 8 | The laugh detector rolls in step by step; key presses stay as the backup. |
| [D-61](agent/decisions-detail.md#d-61) | Nov 8 | Demo shows check the AI judge and the trained model; training on shows waits until about 30 shows. |
| [D-62](agent/decisions-detail.md#d-62) | Nov 8 | Small PRs, each reviewed by a second AI before the user sees it. |
| [D-63](agent/decisions-detail.md#d-63) | Nov 8 | Safety filter starts with word rules; add an AI check only if the rules miss. |
| [D-64](agent/decisions-detail.md#d-64) | Nov 8 | Scraps from the Loft allowed us to use its transcripts for this project, locally. |
| [D-65](agent/decisions-detail.md#d-65) | Nov 8 | Split the decisions notebook into one per project area. |
| [D-66](agent/decisions-detail.md#d-66) | Nov 8 | The hand-written joke bank gets bit structure: opener, tags, closer. |
| [D-67](agent/decisions-detail.md#d-67) | Nov 8 | Exploring 20% of the time costs under 1 laugh a show in 5 of 6 simulated rooms; the extreme room misses. |
| [D-68](agent/decisions-detail.md#d-68) | Nov 8 | Keep 20% exploration; the extreme room no longer counts against the bar. |
| [D-69](agent/decisions-detail.md#d-69) | Nov 8 | Word rules alone missed too much; add an AI check for each bit. |
| [D-70](agent/decisions-detail.md#d-70) | Nov 8 | Rules fixed and AI check added; a line is blocked whenever the check can't confirm it. |
| [D-71](agent/decisions-detail.md#d-71) | Nov 8 | The safety check may take up to 3 s (was 2 s); it runs ahead, so the audience doesn't wait. |
| [D-72](agent/decisions-detail.md#d-72) | Nov 8 | Safety bar: catch at least 75% of new, indirect offensive lines. No human pre-approves scripts. |
| [D-73](agent/decisions-detail.md#d-73) | Nov 8 | The safety filter passed on a second fresh set (24 of 25). |
| [D-74](agent/decisions-detail.md#d-74) | Nov 14 | Generator: one AI call per bit, next line prepared while one plays, filler lines between bits, hand-written jokes as backup. |
| [D-75](agent/decisions-detail.md#d-75) | Nov 14 | Backup jokes go through the word rules only, so they still work when the AI service is down. |
| [D-76](agent/decisions-detail.md#d-76) | Nov 14 | A bit keeps one storyline across its jokes, so the closer can call back to the opener. |
| [D-77](agent/decisions-detail.md#d-77) | Nov 14 | Filler lines are logged too (log format version 2). |
| [D-78](agent/decisions-detail.md#d-78) | Nov 14 | Generated shows meet the timing bars: no noticeable dead air. |
| [D-79](agent/decisions-detail.md#d-79) | Nov 15 | Room audio is recorded through the browser, only when consent is ticked for that show. |
| [D-80](agent/decisions-detail.md#d-80) | Nov 15 | Audio timing bar changed: a steady 0.24 s delay is harmless; only jitter and drift count. Both passed. |
| [D-81](agent/decisions-detail.md#d-81) | Nov 15 | Operator screen: full-width waveform, lilac theme; it never shows the chosen comedy style. |
| [D-82](agent/decisions-detail.md#d-82) | Nov 15 | The first live show uses Gemini to write jokes, so the full system gets tested. |
| [D-83](agent/decisions-detail.md#d-83) | Nov 15 | After each show the operator enters the room's 1-5 rating; a script scores the acceptance run. |
| [D-84](agent/decisions-detail.md#d-84) | Nov 15 | Review fixes to the rating prompt and the scoring script. |
| [D-85](agent/decisions-detail.md#d-85) | Nov 15 | Acceptance run passed: 3 simulated shows and 1 live show ran clean. Stage 0 is done. |
| [D-86](agent/decisions-detail.md#d-86) | Nov 24 | Two decision files: this one for people, a detailed one for agents. |
| [D-87](agent/decisions-detail.md#d-87) | Nov 24 | Docs people read stay short and plain; agent-only detail lives in `docs/agent/`. |
| [D-88](agent/decisions-detail.md#d-88) | Nov 15 | The safety check gets more time (25 s) and one retry on a garbled reply, so one slow reply no longer sends a whole bit to the backup jokes. |
| [D-89](agent/decisions-detail.md#d-89) | Nov 24 | Tag the 90k rJokes jokes with the room each suits (venue, energy, age) for Tingyu's judge; upvotes ignore the room, so a gain is not expected. |
| [D-90](agent/decisions-detail.md#d-90) | Nov 24 | A paused or cut-short show still trains the director and judge from its finished parts; pauses get log rows, and its rating isn't reported. |
| [D-91](agent/decisions-detail.md#d-91) | Nov 24 | Shows open with the best bits from earlier live shows, so the room never hears 10 s of silence. Teen rooms use only bits from teen shows. Not yet measured. |
| [D-92](agent/decisions-detail.md#d-92) | Nov 24 | Compare Gemini Flash-Lite, Gemini Flash and Claude Haiku 5.5 for writing bits on 30 prompts with your blind ranking; the bar was fixed before the run: a model must be fast, rarely time out and rarely fail, and a model with no safe thread is ranked last. |

New decision: add its full entry to the detailed log with the next ID, then a row here in the same PR.
A test fails if the two files' IDs don't match.
