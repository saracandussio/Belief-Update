"""Inspect how the model's chat template wraps the search tool.

Prints:
- the prompt with the search tool declared (what the agent sees at step 0);
- the tool-response wrapper read off the chat template, compared with the
  built-in fallback in belief_update/utils.py.
The search stage uses the wrapper derived here; the fallback is used only if
derivation fails. Run once per model family before generating.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from belief_update.modeling import build_prompt_ids, derive_response_template
from belief_update.utils import PROMPTS, TOOL_FORMATS, get_tool_format


def main(model_name: str = "Qwen/Qwen3.8-27B", tool_format: Optional[str] = None,
         chat_template_kwargs: Optional[dict] = None):
    from transformers import AutoTokenizer

    fmt_name = get_tool_format(model_name, tool_format)
    tok = AutoTokenizer.from_pretrained(model_name)
    native = fmt_name != "searchr1"
    ids = build_prompt_ids(tok, PROMPTS["search_native" if native else "search_searchr1"]
                           .format(question="Q?"), model_name, with_search_tool=native,
                           chat_template_kwargs=chat_template_kwargs)
    print(f"== tool format: {fmt_name}\n== prompt at step 0:\n{tok.decode(ids)}\n")

    derived, json_encoded = derive_response_template(tok, fmt_name, model_name,
                                                     chat_template_kwargs)
    fallback = TOOL_FORMATS[fmt_name]["response_template"]
    print(f"== derived wrapper (json_encoded={json_encoded}):\n{derived!r}")
    print(f"== built-in fallback:\n{fallback!r}")
    if derived is None:
        print("Could not derive the wrapper: the search stage will use the fallback. "
              "Check that the fallback matches the template above.")
        sys.exit(1)
    print("Same as fallback." if derived == fallback else
          "Differs from fallback: the search stage will use the derived one.")


if __name__ == "__main__":
    fire.Fire(main)
