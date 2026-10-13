# PRV demo: Plan-Retrieve-Verify on synthetic toy videos

A minimal, offline demo of the method in *Plan-Retrieve-Verify: An LLM-Driven Framework for Long-Video Audio-Visual Evidence Reasoning* (ACM Multimedia 2026).

> **Notice.** This is a minimal demo that illustrates the method. It is **not** the code used for the paper's experiments, and it does not reproduce any number reported in the paper. All videos, questions and outputs are synthetic. The controller is a rule-based mock, not an LLM.

## The paper in one paragraph

Questions about long videos need evidence from visual scenes, speech and environmental audio. Single-round retrieval followed by answer generation never plans what evidence to seek and never checks whether the evidence supports the conclusion. PRV treats the task as iterative evidence acquisition. An LLM controller writes an evidence plan (which modalities and temporal regions to query), then calls six tools (visual retriever, audio retriever, ASR parser, OCR extractor, temporal checker, contradiction verifier) in a Plan, Retrieve, Verify, Counter-Retrieve loop. Evidence is kept in a workspace with support and contradiction edges. The controller outputs an answer with a confidence, and abstains when evidence is insufficient or contradictory. Training adds counterfactual tool-use trace supervision with five perturbation types (audio-swap, visual-swap, temporal-shuffle, same-entity distractor, pseudo-support), so that planning, tool selection and verification decisions are supervised, not only the final answer.

## Quickstart

Python 3.10 or newer. The demo uses only the standard library; `pytest` is needed for the tests.

```bash
python -m prv.demo                          # trajectories + 5 supervision records
python -m prv.demo --out records.jsonl      # also write the records as JSONL
python -m pytest -q                         # tests
```

The demo runs in well under a second and needs no network or API key.

## What the demo shows

`python -m prv.demo` runs three questions, each on its own synthetic video, and prints the trajectory:

1. **Verified answer.** Speech and on-screen text agree (support edge), so the answer is returned with its evidence.
2. **Counter-retrieval.** Vision and speech contradict. The controller searches the whole video for counter-evidence, finds on-screen text that supports vision, and the speech claim is outvoted.
3. **Abstention.** Speech and on-screen text contradict and nothing resolves the conflict, so the controller abstains.

It then takes the trajectory of case 2, applies the five perturbations, re-runs the controller on each perturbed video, and prints five supervision records. The full records are in `examples/supervision_records.jsonl`.

| Demo component | File | Paper section |
| --- | --- | --- |
| Evidence plan (tool priority vector, temporal region, h0, need_counter) | `prv/llm.py` (`MockController.plan`) | 3.2 |
| Six tools over the JSON segments | `prv/tools.py` | 3.3 |
| Controller interface (`LLMClient`) and rule-based mock | `prv/llm.py` | 3.2, 3.4 |
| Closed loop, step budget, stop signal, trajectory | `prv/loop.py` | 3.4 (Algorithm 1) |
| Support/contradiction workspace | `prv/workspace.py` | 3.5 |
| Calibrated answer and abstention below a threshold | `prv/llm.py` (`MockController.synthesize`) | 3.6 (Inference) |
| Five perturbations and supervision records | `prv/counterfactual.py` | 3.6 |

### Supervision record

Each JSONL record holds the perturbation edits, the decision labels on the clean video (`original_labels`), the labels on the perturbed video (`target_labels`), and which decisions changed. Labels cover the decisions the paper says are supervised:

- `plan`: needed modalities, temporal region, need_counter.
- `tool_choice`: the sequence of tool calls with their regions.
- `verification`: support / contradict / neutral label per evidence pair, temporal relations, outvoted evidence, unresolved contradictions.
- `final`: answer or abstention.

`original_trace_reproduced` is `true` when the perturbed run keeps the same tool calls and the same final decision. The paper penalises reproducing the original trace on corrupted input (its trace loss); this demo only emits the labels and does not compute any loss.

## Repository layout

```
prv/
  data.py            segment / video / question types, loader, coarse summary
  data/toy_videos.json   3 synthetic videos (8 segments each) and 3 questions
  text.py            lexical helpers (tokens, polar word pairs, claim flipping)
  tools.py           the six tools
  workspace.py       evidence graph, outvoting rule, unresolved contradictions
  llm.py             LLMClient protocol, Plan/ToolCall/Answer, MockController
  loop.py            run_prv() and trajectory printing
  counterfactual.py  five perturbations, decision labels, supervision records
  demo.py            python -m prv.demo
tests/               pytest: each tool, abstention, each perturbation type
examples/            supervision_records.jsonl (output of the demo)
```

## Simplifications

- **Synthetic data.** Three toy videos, each eight 10-second segments of JSON text fields (visual caption, ASR text, audio events, on-screen OCR text). No media, no real dataset.
- **Mock controller.** `MockController` replaces the LLM with fixed rules: modality cues in the question set the tool priority vector; the temporal region comes from word overlap between the question and a coarse summary; yes/no questions set `need_counter`. The paper makes the plan in one structured LLM call.
- **Coarse summary.** One caption plus audio events per segment. The paper samples a keyframe every 10 s, captions it, and appends audio tags.
- **Tools are lexical.** Retrievers rank by word overlap, ASR and OCR read stored text, and the contradiction verifier is a rule over shared words, polar word pairs (running / shut, open / closed) and numbers. The paper uses InternVideo2, BEATs, Whisper, an OCR model and a learned bilinear polarity scorer. The thresholds for support and contradiction are not used.
- **Verification rule.** A contradiction is settled by counting modalities on each side (the side with more modalities wins; a tie stays unresolved). The paper leaves this judgement to the LLM.
- **Counter-retrieval policy.** The mock tries an unqueried modality over the whole video, using the disputed texts as the query. The verifier is called by the loop on every new evidence pair, not chosen by the controller.
- **Confidence.** A toy formula (more agreeing modalities give higher confidence; an unresolved contradiction divides it by five). It is not calibrated. Defaults `max_steps=8` and `tau_r=0.3` follow the paper's stated settings.
- **Workspace.** Support, contradict and temporal edges only. The coreference edge and fused multimodal features are omitted.
- **Perturbations.** The paper names the five types but this demo defines how each acts on the toy JSON: audio-swap replaces speech and audio events of the evidence segments with a donor video's; visual-swap replaces captions; temporal-shuffle reverses segment contents over fixed timestamps; the same-entity distractor adds a neutral line naming the same entity; pseudo-support adds a fabricated line asserting the opposite claim. Whether a perturbation changes any decision depends on the retrieval region, so some records show no change.
- **Supervision targets.** Targets come from re-running the mock controller on the perturbed video. The paper builds ground-truth traces with an oracle planner that has ground-truth evidence spans, then checks a sample by hand.
- **No training, no metrics.** No loss, model, benchmark or evaluation is included. There is no `--llm` backend hook; any object with the four `LLMClient` methods can be passed to `run_prv`.

## Citation

The DOI is not yet available. Update this entry when it is registered.

```bibtex
@inproceedings{xiao2026prv,
  title     = {Plan-Retrieve-Verify: An LLM-Driven Framework for Long-Video Audio-Visual Evidence Reasoning},
  author    = {Xiao, Yifan and Xu, Dexuan and Li, Shijie and Huang, Yu},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil}
}
```

## License

MIT. See `LICENSE`.
