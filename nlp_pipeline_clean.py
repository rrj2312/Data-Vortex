import re, html

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def clean_text(s, strip_user=True, keep_hashtag_word=True, unescape=True):
    """Deterministic, reversible-by-flag cleaning. Each step is ablated below."""
    if unescape:
        s = html.unescape(html.unescape(s))          # double-encoded entities seen in Dataset 1
    s = s.replace('"""', ' ').replace('""', ' ')      # CSV over-quoting artifact
    s = URL_RE.sub(" URLTOKEN ", s)
    if strip_user:
        s = USER_RE.sub(" USERTOKEN ", s)
    if keep_hashtag_word:
        s = HASH_RE.sub(r" \1 ", s)                   # #Barcelona -> Barcelona (keeps semantics)
    s = WS_RE.sub(" ", s).strip()
    return s


