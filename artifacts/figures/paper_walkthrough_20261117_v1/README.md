# Retained-sample paper illustration

Figure 3 of the English paper uses the first retained trip of cache-046, cache-046_day7, over 0--180 s. At the user's request, its visibly multi-turn geometry replaces the nearly straight cache-029 example. Selection inspected only GPS shape among retained TEST trips, not attack or utility scores. This is an illustrative case, not a representative-performance claim; no defense or attacker was rerun.

- Left: all five Q, recorded GPS and internal reference Z over the actual benchmark SUMO/OSM road network. The inset enlarges the GPS/Z footprint; the main map includes a 1 km scale and north arrow.
- Map geometry comes from the first passenger lane of each external SUMO edge, with the same projection/origin as all sample points. Display geometry is rounded to 1 mm. Open Q markers show 0 s and triangles show 180 s; intermediate recorded coordinates are dots. No straight line is claimed as a reconstructed route.
- Right: selected ticks, protected-read/refresh events, belief operations, cumulative Geo-I spending, nominal Q allowance reserved and received-POI cache size.
- The 20/40 s steps do not perform a protected read; true GPS still changes and stays available for local ranking. The refreshes at 60/120/180 s each incur two units; spending reaches 7 units at 180 s.
- The first three frames belong to the same committed segment. This sample's first three sets of five coordinates coincide.

The belief row shows which operation was triggered, not reconstructed posterior probabilities. The map illustrates retained coordinates, not a noise-distance or privacy guarantee. Geo-I spending has units m^-1; Q allowance is dimensionless. Client diagnostics do not become attacker features.

Regenerate with:

    python experiments/export_paper_walkthrough_20261117.py
    python experiments/export_paper_walkthrough_20261117.py --check

The JSON data and receipt bind the figure to both retained benchmark artifacts, the public map/projection and the exporter. The road subset is rebuilt from the pinned network and authenticated by its digest and edge count, rather than duplicated in the sample JSON. Neither benchmark artifact is modified.
