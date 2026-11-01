from bangla_datasets.tools.catalog import ALL_TOOLS, DOMAIN_TOOLS


def test_catalog_has_18_domains() -> None:
    assert len(DOMAIN_TOOLS) == 18


def test_catalog_has_at_least_50_tools() -> None:
    assert len(ALL_TOOLS) >= 50


def test_all_tool_names_ascii() -> None:
    for t in ALL_TOOLS:
        assert t.name.isascii(), f"non-ASCII tool name: {t.name}"


def test_all_tool_param_keys_ascii() -> None:
    for t in ALL_TOOLS:

        def keys(node: object) -> object:
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k
                    yield from keys(v)

        for k in keys(t.parameters_json_schema):
            if isinstance(k, str):
                assert k.isascii(), f"non-ASCII key in {t.name}: {k}"
