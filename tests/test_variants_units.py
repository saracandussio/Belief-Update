from belief_update.units import build_units
from belief_update.variants import build_variants_prompt, validate_variants

ITEM = {"id": "1", "question": "What is the capital of the state where Dallas is?",
        "golden_answers": ["Austin"], "metadata": {"type": "compositional"}}
OUT = {
    "chain": True,
    "hops": [{"question": "In which state is Dallas", "answer": "Texas"},
             {"question": "What is the capital of Texas?", "answer": "Austin"}],
    "paraphrases": ["Which city is the capital of the state that contains Dallas?",
                    "What is the capital of Texas, home of Dallas?",
                    "Name the capital city of the US state in which Dallas lies."],
    "hop_paraphrases": [["Dallas is in which state?"], ["Texas's capital is which city?"]],
}


def test_validate_drops_leaky_paraphrase():
    rec = validate_variants(ITEM, OUT, k=3)
    assert rec["valid"]
    assert rec["hops"][0]["question"].endswith("?")
    assert len(rec["paraphrases"]) == 2  # the one revealing "Texas" is dropped
    assert any("leaking" in i for i in rec["issues"])


def test_validate_flags_wrong_last_hop_and_native_override():
    bad = {**OUT, "hops": [OUT["hops"][0], {"question": "Capital of Texas?", "answer": "Houston"}]}
    assert not validate_variants(ITEM, bad, k=3)["valid"]
    native = {**ITEM, "metadata": {"question_decomposition": [
        {"question": "Dallas >> located in state", "answer": "Texas"},
        {"question": "capital of #1", "answer": "Austin"}]}}
    assert "Reference chain" in build_variants_prompt(native, 3)
    rec = validate_variants(native, OUT, k=3)
    assert rec["hops_source"] == "native" and rec["valid"]


def test_build_units():
    rec = validate_variants(ITEM, OUT, k=3)
    variants = {"1": rec}
    q_units = build_units([ITEM], variants, "question")
    assert [u["uid"] for u in q_units] == ["1|q|orig", "1|q|para0", "1|q|para1"]
    h_units = build_units([ITEM], variants, "hops")
    assert [u["uid"] for u in h_units] == ["1|h0|orig", "1|h0|para0", "1|h1|orig", "1|h1|para0"]
    assert h_units[0]["golds"] == ["Texas"]
    assert build_units([ITEM], None, "hops") == []
    assert len(build_units([ITEM], None, "question")) == 1


def test_supporting_sentences_both_layouts():
    from belief_update.utils import supporting_sentences
    dict_layout = {"metadata": {
        "context": {"title": ["Dallas", "Texas"],
                    "sentences": [["Dallas is a city.", "It is in Texas."], ["Austin is the capital."]]},
        "supporting_facts": {"title": ["Dallas", "Texas"], "sent_id": [1, 0]}}}
    list_layout = {"metadata": {
        "context": [["Dallas", ["Dallas is a city.", "It is in Texas."]], ["Texas", ["Austin is the capital."]]],
        "supporting_facts": [["Dallas", 1], ["Texas", 0], ["Missing", 0], ["Texas", 9]]}}
    expected = ["Dallas: It is in Texas.", "Texas: Austin is the capital."]
    assert supporting_sentences(dict_layout) == expected
    assert supporting_sentences(list_layout) == expected
    assert supporting_sentences({"metadata": {"type": "x"}}) == []
    item = {**ITEM, "metadata": dict_layout["metadata"]}
    prompt = build_variants_prompt(item, 3)
    assert "Gold supporting sentences" in prompt and "Austin is the capital." in prompt
    assert validate_variants(item, OUT, k=3)["hops_source"] == "llm+support"
