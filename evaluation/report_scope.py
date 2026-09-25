"""Current reporting scope; frozen datasets and historical runs stay intact."""
SCENARIOS = ('S1', 'S2', 'S3', 'S9', 'S10')
EXCLUDED_CASES = frozenset({'S10.B'})


def suffixes(scenario):
    return 'AC' if scenario == 'S10' else 'ABC'


PRIORITY_CASES = tuple(f'{s}.{c}' for s in SCENARIOS for c in suffixes(s))
ALL_CASES = tuple(f'S{s}.{c}' for s in range(1, 11) for c in suffixes(f'S{s}'))


def scenario_macro(values):
    """Each scenario gets equal weight, its available cases equal weight.

    Missing values remain partial descriptive evidence (e.g. offline reference),
    not complete-scope scores. Callers must preserve completeness labels.
    """
    means = []
    for scenario in SCENARIOS:
        v = [values.get(f'{scenario}.{c}') for c in suffixes(scenario)]
        v = [x for x in v if x is not None]
        if v:
            means.append(sum(v)/len(v))
    return sum(means)/len(means) if means else None
