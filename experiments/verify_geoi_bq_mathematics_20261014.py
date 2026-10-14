"""Recompute supplemental exact mathematical certificates; never score a model.

Run without --check to write the separate supplement receipt. --check compares
its entire deterministic contents, including source hashes and the frozen main
research manifest. No GPS/cohort data, private ledger, or RNG key is loaded.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from benchmark.bounded_inference import inference_certificate
from benchmark.finite_session_bounds import all_duration_ratio, posterior_mass_bound
from benchmark.geo_i_event_bound import event_log_bound
from benchmark.reference_utility_transfer import reference_transfer
from benchmark.whole_path_chi_square import whole_path_bound
from core.research_release import load_release


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'docs/reviews/2026-10-14_mathematical_completion.json'
SOURCES = (
    'docs/research/2026-10-14_geoi_bq_mathematical_foundation.md',
    'benchmark/reference_utility_transfer.py',
    'benchmark/conditional_gain.py',
    'benchmark/finite_query_kernel.py',
    'benchmark/finite_session_bounds.py',
    'benchmark/whole_path_chi_square.py',
    'benchmark/session_utility_bounds.py',
    'benchmark/bounded_inference.py',
    'benchmark/geo_i_event_bound.py',
    'core/bounded_query_session.py',
    'configs/geo_i_bq_v1.json',
    'experiments/verify_geoi_bq_mathematics_20261014.py',
    'tests/test_geoi_bq_proof_chain.py',
    'tests/test_reference_utility_transfer.py',
    'tests/test_conditional_gain.py',
    'tests/test_whole_path_chi_square.py',
    'tests/test_bounded_inference.py',
    'tests/test_geo_i_event_bound.py',
    'tests/test_finite_session_bounds.py',
)


def certificate():
    release = load_release()  # Authenticates the pre-existing main reference.
    ratio = all_duration_ratio(1)
    path = whole_path_bound(1., 64, infinite=True)
    inference = inference_certificate()
    assert ratio < F(2437372, 1000000)
    assert F(path['exact_rational_utility_shift_upper']) < F(553, 10000)
    assert F(inference['exact_information_bits_upper']) < F(18, 1000)
    unknown = reference_transfer([F(99, 100)])
    assert unknown['session_expected_recall_lower'] is None
    return dict(
        schema='geoi-bq-mathematical-supplement-v1',
        status='exact numeric certificates pass; frozen main manifest authenticates',
        symbolic_proofs='derived in linked manuscript; finite oracles do not substitute for proofs',
        benchmark_or_algorithm_changed=False,
        main_release_pinned_file_count=len(release['files_sha256']),
        sources_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
        single_session_gamma1=dict(
            all_duration_likelihood_ratio_upper=str(ratio),
            balanced_prior_posterior_upper=str(posterior_mass_bound(F(1, 2), ratio)),
            utility=path, inference=inference,
            first_fix_10m_ideal_geoi_log_bound=str(event_log_bound([10, 0])),
            later_fix_10m_ideal_geoi_log_bound=str(event_log_bound([0, 10])),
        ),
        actual_cohort_public_reference_transfer=unknown,
        illustration_not_cohort_estimate=reference_transfer(
            [F(99, 100)], conditional_tv_bounds=[F(1, 10)],
            proxy_error_bounds=[F(1, 20)], recall_threshold=F(4, 5)),
        finite_oracle_files=['tests/test_geoi_bq_proof_chain.py',
                             'tests/test_reference_utility_transfer.py'],
        reproducible_checks='pytest the two oracle files and the existing kernel/path/inference/ledger tests',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = certificate()
    if args.check:
        if json.loads(OUTPUT.read_text()) != result:
            raise SystemExit('Mathematical supplement receipt changed; inspect and regenerate explicitly')
    else:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('PASS: rational privacy, inference and utility certificates; main reference unchanged')


if __name__ == '__main__':
    main()
