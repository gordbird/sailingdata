import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("sailingheatmap.py")
spec = importlib.util.spec_from_file_location("sailingheatmap", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_activity_metadata_uses_iso_timestamp_for_label_and_filename_for_tooltip():
    tcx_file = Path(__file__).parent / "tcx" / "2025" / "activity_19877400725.tcx"

    assert module.read_activity_timestamp(tcx_file) == "2025-07-28T18:45:05.000Z"
    assert module.activity_name_for_tooltip(tcx_file) == "activity_19877400725"
