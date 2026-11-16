# Practical pipeline

## 1. Freeze the items (`prepare_data.py`)

Loads `RUC-NLPIR/FlashRAG_datasets`, using the test split if present and dev
otherwise (same rule as Search-R1). Questions are normalized as in Search-R1
(strip, trailing `?`). With `--max_questions`, a seeded random subset is taken.
The first N would not do, because files can be ordered by question type.
The script prints the metadata keys of the first item: **check them on the
first run** (see caveats).

## 2. Hop chains and paraphrases (`make_variants.py`)

A stronger model receives the question, the gold answers and, when available,
the dataset's own chain (MuSiQue `question_decomposition`, with `#k`
placeholders resolved to gold answers) or evidence triples (2Wiki `evidences`).
It returns:

- `chain`: whether the question is a sequential chain or a comparison;
- `hops`: standalone single-hop questions with short answers;
- `paraphrases`: rewordings of the full question that keep bridges implicit;
- `hop_paraphrases`.

Validation (`belief_update/variants.py`):

- 1 to 4 hops, no placeholders, non-empty answers;
- with a native chain: same number of hops, and **native answers always
  override** the writer's;
- for chains, the last hop answer must match a gold answer;
- paraphrases that contain an intermediate answer absent from the original
  question are dropped, because they would leak the bridge.

Items failing structural checks get `valid: false` and are skipped by the
hop stage. Their original question is still used.

## 3. Closed-book (`generate_closed_book.py`)

Prompt (`PROMPTS["closed_book"]`): answer from own knowledge, final answer in
`<answer>` tags. Two passes:

- **CoT**: thinking on;
- **no-CoT**: thinking off through the chat template switch
  (`enable_thinking=False`). No suffix is appended; models without the switch
  skip this pass. gpt-oss is one of them: its reasoning cannot be turned off,
  only shortened with `reasoning_effort`. Commitment-Boundary forced an empty
  analysis channel plus an answer suffix for it, which this repo avoids by design.

`--target question` runs the composed question; `--target hops` runs every hop.
Both include paraphrases by default. The answer is the last `<answer>` after
the end-of-thinking marker. Records keep prompt and trace token ids for later
replay.

## 4. Search agent (`generate_search_traces.py`)

At start-up the script renders a dummy [user, tool call, tool result]
conversation with the model's chat template and keeps the text between the end
of the call and the next assistant turn as the tool-response wrapper. For
gpt-oss the template JSON-encodes tool outputs, and so does the loop. For
Qwen3.8 the wrapper includes the `<think>\n` that the template opens for the
next turn.

Each of the `num_out` samples is an independent trajectory. At each step:

1. generate until the tool-call closing tag or the end of the turn;
2. if the turn closes a well-formed call, query the retriever, append the
   tool response, and continue;
3. if it gives an `<answer>`, stop.

`finish_reason` is one of `answer`, `no_answer`, `invalid_call`, `max_calls`,
`length`, `context`. There is no forced final answer when the budget runs out.

## 5. Judge (`judge_answers.py`, optional)

Only predictions that are not EM-correct are judged, once per distinct
(question, golds, prediction). Labels: `correct` / `incorrect`. Validate on a
hand-labelled sample before using `--metric judge`.

## 6. Analysis (`analyze.py`)

Pass rate per unit is the fraction of samples judged correct. The item table
holds:

- pass rates per item: question (orig / paraphrases, CoT / no-CoT), each hop, search;
- mean number of search calls and finish reasons;
- `category`, `format_sensitive` and `prior_defined` (see README).

The summary adds:

- category counts;
- the closed-book × search grid (`cb_known/search_ok` etc.: how often search
  only confirms what the model already knew);
- per-type breakdown (2Wiki/HotpotQA types).

## Caveats to check on the first real run

- **FlashRAG metadata fields** are read defensively (`question_decomposition`,
  `evidences`, `type`). This was written without access to the hub, so check
  the printed metadata keys. If MuSiQue's chain sits under a different key,
  fix `native_decomposition` in `utils.py`.
- **Tool-response wrapper**: run `scripts/check_template.py` for each new
  model family and read the step-0 prompt and the derived wrapper. If
  derivation fails, the search stage falls back to the built-in string and
  prints a warning.
- **Qwen3.8 `reasoning_effort`**: the model card lists it (default `xhigh`), but
  it was not verified as a chat-template kwarg. Check that it changes the
  rendered prompt in `check_template.py` before relying on it.
- **Hop answers have no aliases** (single gold string), so hop EM is strict.
  Use the judge before trusting `hops_known` / `shortcut`.
- **Comparison questions** (`chain: false`, 2Wiki `comparison`, HotpotQA
  yes/no) have no bridge chain. Keep them out of the chain analyses.
- **Wikipedia 2018 corpus**: for facts that changed after 2018, retrieval can
  contradict a correct prior. These are natural conflicts; flag them before
  the conflict experiments.
