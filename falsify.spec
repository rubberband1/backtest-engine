# PyInstaller build of the desktop application.
#
#     pip install -e ".[build]"
#     cd ui && npm ci && npm run build && cd ..
#     python -m scripts.make_icons
#     pyinstaller falsify.spec --noconfirm
#
# The result is dist/Falsify/Falsify.exe, which starts by double click.
#
# One directory rather than one file, deliberately. A onefile build unpacks
# pandas, numpy, scipy and pyarrow into a temporary directory on every launch,
# which costs ten to twenty seconds each time and buys a tidier download. The
# executable inside this folder is still what somebody double-clicks.
#
# Bundled: the built dashboard, the icon, the strategy library and the
# synthetic dataset. Those last two make a fresh install a working
# application rather than an empty one - `desktop.py` copies the strategies
# out beside the executable on first run so they can be edited and kept.
# Everything a run produces (runs/, logs/, data_cache/) is written next to the
# executable, never into the bundle, which a onedir build shares between users
# and a onefile build deletes on exit.

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "ui" / "dist"), "ui/dist"),
    (str(ROOT / "brand" / "falsify.ico"), "brand"),
    (str(ROOT / "strategies"), "strategies"),
    (str(ROOT / "fixtures" / "data_cache"), "fixtures/data_cache"),
]

# uvicorn picks its event loop and its HTTP and websocket protocols by
# importing them by name at runtime, so nothing static references them and
# the analysis cannot see them.
hiddenimports = [
    *collect_submodules("uvicorn"),
    "anyio._backends._asyncio",
]

analysis = Analysis(
    ["desktop.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Weight, not correctness: none of these is imported by the application,
    # and matplotlib alone is forty megabytes of nothing.
    excludes=["tkinter", "matplotlib", "IPython", "notebook", "pytest", "mypy"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Falsify",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console window behind the application window. Anything worth saying
    # when the backend fails is said in the window itself, by failure_page().
    console=False,
    icon=str(ROOT / "brand" / "falsify.ico"),
)

COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="Falsify",
)
