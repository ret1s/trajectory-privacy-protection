"""Report labels, separate from immutable experiment case identifiers."""

SOURCE_TO_REPORT = {'S10.C': 'S10.B'}


def display_case(source_case):
    return SOURCE_TO_REPORT.get(source_case, source_case)


def report_label_manifest():
    return {
        'report_to_source': {v: k for k, v in SOURCE_TO_REPORT.items()},
        'unchanged_case_labels': 'identity mapping',
        'policy': 'Presentation labels only; retain source records, metrics and experiment identifiers.',
    }
