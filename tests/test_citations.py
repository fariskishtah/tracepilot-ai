from src.agents.synthesis import _clean, abstention_message, attribute_uncited
from src.citations.validator import ids_in, numbers_supported, sentence_supported, strip_cites, validate_answer
from src.tools.base import EvidenceItem


def ev(i, text, kind="document"):
    return EvidenceItem(i, kind, "t", text)


EVID = {"S1": ev("S1", "Overfitting happens when a model memorizes noise: training error is very low but validation error is much higher."),
        "D1": ev("D1", "[product_prices] mean of 'price_usd' = 105.041 (rows used: 40 of 40)", "dataset")}


def test_id_extraction():
    assert ids_in("a [S1] b [S2, D1]") == ["S1", "S2", "D1"] and strip_cites("Fact [S1].") == "Fact."


def test_invalid_ids_removed_and_reported():
    assert "[S9]" not in _clean("Overfitting is bad [S9]. Another [S1].", EVID)
    r = validate_answer("Overfitting means high validation error [S9].", EVID)
    assert r.invalid_ids == ["S9"] and r.n_supported == 0                                   # syntax alone is not enough


def test_support_checked_not_just_syntax():
    good = validate_answer("Overfitting happens when training error is very low and validation error is much higher [S1].", EVID)
    bad = validate_answer("Overfitting is when the GPU runs out of memory during inference [S1].", EVID)
    assert good.n_supported == 1 and bad.n_supported == 0 and bad.unsupported()[0].reason


def test_numbers_must_be_in_evidence():
    assert numbers_supported("The mean price is 105.04", EVID["D1"].text) == []
    assert numbers_supported("The mean price is 250.5", EVID["D1"].text) == ["250.5"]
    r = validate_answer("The mean price is 250.5 [D1].", EVID)
    assert r.n_supported == 0 and "numbers not in evidence" in r.unsupported()[0].reason


def test_uncited_sentences_get_verified_attribution_or_are_dropped():
    txt = "Overfitting happens when training error is very low but validation error is much higher. The moon is made of cheese and orbits Mars daily."
    out, att, drop = attribute_uncited(txt, EVID, ["S1", "D1"])
    assert "[S1]" in out and "moon" not in out and att == 1 and drop == 1


def test_generated_summary_not_used_as_primary():
    from src.tools.base import EvidenceStore
    st = EvidenceStore()
    g = st.add_summary("summary text", ["S1"])
    assert g.kind == "summary" and g not in st.primary()


def test_abstention_message_is_honest():
    m = abstention_message(["capital France"], "Doc (p.1, cosine 0.09)", [])
    assert "could not find" in m.lower() and "France" in m
