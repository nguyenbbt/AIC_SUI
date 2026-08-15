from scripts.mgpux_qwen_proxy import is_authorized


def test_proxy_accepts_only_the_exact_bearer_token():
    assert is_authorized("Bearer private-key", "private-key") is True
    assert is_authorized("Bearer wrong-key", "private-key") is False
    assert is_authorized("Basic private-key", "private-key") is False
    assert is_authorized(None, "private-key") is False


def test_proxy_rejects_an_empty_configured_key():
    assert is_authorized("Bearer anything", "") is False
