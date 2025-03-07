from rl_the_spire.utils.import_module_with_syspath import import_module_with_syspath


def test_slay_the_text() -> None:
    import_module_with_syspath("main", "external/slaythetext")
