"""Assertions shared by isolated SDK clients and the regular test suite."""
import json
import inspect
from pathlib import Path


def normalize(value, fixtures):
    """Normalize only fixture paths and JSON text whitespace, not payload values."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "text":
                try:
                    item = json.loads(item)
                except (ValueError, TypeError):
                    pass
            result[key] = normalize(item, fixtures)
        return result
    if isinstance(value, list):
        return [normalize(item, fixtures) for item in value]
    if isinstance(value, str):
        value = value.replace(str(fixtures.resolve()), "<fixtures>")
        return value.replace("<fixtures>\\", "<fixtures>/")
    return value


def verify_contract(data, fixtures):
    golden = json.loads((Path(__file__).parent / "fixtures/mcp_v1_contract.json").read_text(encoding="utf-8"))
    def tools_for_comparison(tools):
        # CPython 3.13 began stripping common docstring indentation at compile
        # time. Both SDK 1 and SDK 2 use fn.__doc__; the historical v1 surface
        # already differs by this whitespace across Python versions. Normalize
        # ONLY the top-level description. Names and every schema field stay exact.
        return [{**tool, "description": inspect.cleandoc(tool["description"])} for tool in tools]
    assert tools_for_comparison(data["tools"]) == tools_for_comparison(golden["tools"]), "Tool names, schemas or description content changed"
    for label, expected in golden["calls"].items():
        # These two fields belong to the 2026 protocol envelope. Tool payloads,
        # isError, content blocks and structuredContent must remain equivalent.
        actual = {k: v for k, v in data["calls"][label].items() if k not in ("_meta", "resultType")}
        assert normalize(actual, fixtures) == expected, f"Tool contract changed: {label}"
    missing = data["calls"]["missing_format"]
    assert missing["isError"] is True
    assert "format" in missing["content"][0]["text"]
    assert "Field required" in missing["content"][0]["text"]
    capabilities = data["calls"]["capabilities"]
    assert capabilities["isError"] is False
    capabilities = json.loads(capabilities["content"][0]["text"])
    assert "can_capture_here" in capabilities
    assert "ncu-csv" in {f for backend in capabilities["can_digest"] for f in backend["formats"]}
    suggest = data["calls"]["suggest"]
    assert suggest["isError"] is False
    suggest = json.loads(suggest["content"][0]["text"])
    assert isinstance(suggest["ok"], bool)
    assert (suggest.get("command") if suggest["ok"] else suggest.get("alternative"))
    assert [p["name"] for p in data["prompts"]["prompts"]] == ["perfdigest_usage"]
    assert data["usage"]["messages"]
