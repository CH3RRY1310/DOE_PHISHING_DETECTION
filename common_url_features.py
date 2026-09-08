import re
from urllib.parse import urlparse

import pandas as pd

COMMON_FEATURES = [
    "url_length",
    "domain_length",
    "path_length",
    "subdomain_count",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "digit_count",
    "letter_count",
    "special_char_count",
    "at_count",
    "question_count",
    "equals_count",
    "ampersand_count",
    "slash_count",
    "is_domain_ip",
    "is_https",
    "has_obfuscation",
]

IPV4_PATTERN = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")


def extract_url_features(url):
    text = str(url).strip()
    parsed = urlparse(text if "://" in text else "http://" + text)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    return {
        "url_length": len(text),
        "domain_length": len(hostname),
        "path_length": len(path),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "dot_count": text.count("."),
        "hyphen_count": text.count("-"),
        "underscore_count": text.count("_"),
        "digit_count": sum(character.isdigit() for character in text),
        "letter_count": sum(character.isalpha() for character in text),
        "special_char_count": sum(not character.isalnum() for character in text),
        "at_count": text.count("@"),
        "question_count": text.count("?"),
        "equals_count": text.count("="),
        "ampersand_count": text.count("&"),
        "slash_count": text.count("/"),
        "is_domain_ip": int(bool(IPV4_PATTERN.fullmatch(hostname))),
        "is_https": int(parsed.scheme.lower() == "https"),
        "has_obfuscation": int("%" in text or "//" in text[8:]),
    }


def extract_features(urls):
    return pd.DataFrame(
        [extract_url_features(url) for url in urls],
        columns=COMMON_FEATURES,
    )
