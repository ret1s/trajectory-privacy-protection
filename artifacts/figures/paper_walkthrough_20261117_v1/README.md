# Retained-sample paper illustration

Figure 3 of the English paper uses the first trip of the fixed first TEST family, cache-029_day7, over a fixed 0--180 s prefix. No sample was selected by attack or utility score, and no defense or attacker was rerun.

- Left: east displacement of the recorded GPS, internal reference Z and Q1. All five Q are still sent at each frame.
- Right: selected ticks, protected-read/reuse events, belief operations, cumulative Geo-I spending, nominal Q allowance reserved and received-POI cache size.
- The 20/40 s steps do not perform a protected read; true GPS still changes and stays available for local ranking. Reusing Z at 180 s incurs one private-test unit.
- The first three frames belong to the same committed segment. This sample's first three sets of five coordinates coincide.

The belief row shows which operation was triggered, not reconstructed posterior probabilities. The one-dimensional trace is a display projection, not a noise-distance or privacy guarantee. Geo-I spending has units m^-1; Q allowance is dimensionless. Client diagnostics do not become attacker features.

Regenerate with:

    python experiments/export_paper_walkthrough_20261117.py
    python experiments/export_paper_walkthrough_20261117.py --check

The JSON data and receipt bind the figure to both retained benchmark artifacts, the public map/projection and the exporter. Neither benchmark artifact is modified.
