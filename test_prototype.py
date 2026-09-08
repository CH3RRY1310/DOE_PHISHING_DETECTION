from prototype import URLPhishingPrototype


prototype = URLPhishingPrototype()
cases = [
    ("https://www.southbankmosaics.com", "LEGITIMATE"),
    ("https://www.uni-mainz.de", "LEGITIMATE"),
    ("http://www.teramill.com", "PHISHING"),
    ("http://www.f0519141.xsph.ru", "PHISHING"),
]

for url, expected_prediction in cases:
    result = prototype.predict(url)
    assert result["prediction"] == expected_prediction
    assert 0.0 <= result["phishing_probability"] <= 1.0
    assert len(result["features"]) == 5
    assert len(result["explanation"]) == 5
    print(result["prediction"], f"{result['phishing_probability']:.4f}", url)
