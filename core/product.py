"""What the application is called when it speaks to a person.

The distribution, the Python packages and the repository stay
`backtest-engine`: that name is in import paths, in `pyproject.toml` and in
every clone URL, and renaming it would break all three for a cosmetic gain.
This module is the other half - the name the window title, the dashboard
header and the documentation use - so that the two can differ without either
being written twice.
"""

PRODUCT_NAME = "Falsify"

# Popper: a claim earns its keep only by surviving attempts to break it. The
# engine's whole design - the gates, the trial count, the permutation nulls -
# is a machine for making those attempts, so the tagline is a description
# rather than a slogan.
TAGLINE = "A backtesting engine that tries to prove your strategy wrong."
