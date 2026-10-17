"""Build privacy-only meeting constants/provenance; no benchmark or sampling."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from benchmark.finite_session_bounds import all_duration_ratio, posterior_mass_bound

MANIFEST='configs/geo_i_bq_v1.json'
SOURCES=[MANIFEST,
    'docs/research/2026-10-14_geoi_bq_mathematical_foundation.md',
    'docs/reviews/2026-10-14_mathematical_adversarial_audit.md',
    'docs/reviews/2026-10-14_mathematical_completion.json',
    'docs/research/2026-10-17_continual_geoi_protocol.md',
    'benchmark/finite_session_bounds.py','benchmark/finite_query_kernel.py',
    'benchmark/anchor_belief.py','benchmark/targeted_cache_exploration.py',
    'benchmark/engines/filtered_cover.py','benchmark/engines/matched_filter.py',
    'benchmark/engines/paced_guard.py','core/mechanisms.py',
    'core/query_budget.py','core/bounded_query_session.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decimal_ceiling(value,places):
    scale=10**places
    integer=(value.numerator*scale+value.denominator-1)//value.denominator
    return f'{integer//scale}.{integer%scale:0{places}d}'


def generated():
    manifest=json.loads((ROOT/MANIFEST).read_text())
    for name,digest in manifest['files_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Sealed main source/evidence changed: '+name)
    geo=manifest['parameters']['GeoI']
    B=Fraction(str(geo['nominal_B_per_m']));H=geo['H']
    unit=B/(2*H);cap=(2*H-1)*unit
    assert unit==Fraction(str(geo['u_per_m'])) and cap==Fraction(str(geo['effective_Cs_per_m']))
    upper=all_duration_ratio(1,cutoff=11)
    ratio=decimal_ceiling(upper,6)
    assert upper<Fraction(ratio)
    posterior=decimal_ceiling(100*posterior_mass_bound(Fraction(1,2),upper),2)
    weights=(Fraction(1,2),Fraction(1,2));scores=(1,0);epsilon=Fraction(1,2)
    alpha=epsilon/(2+epsilon);mean=sum(w*z for w,z in zip(weights,scores))
    probabilities=tuple(w*(1+alpha*(z-mean)) for w,z in zip(weights,scores))
    assert probabilities==(Fraction(11,20),Fraction(9,20)) and sum(probabilities)==1
    values={'NominalBudget':str(geo['nominal_B_per_m']),'Horizon':str(H),
            'UnitEpsilon':str(geo['u_per_m']),'GeoCap':str(geo['effective_Cs_per_m']),
            'RatioCeiling':ratio,'PosteriorCeiling':posterior}
    constants='% Generated from sealed config and exact rational bounds; do not edit.\n'
    for name,value in values.items():
        constants+='\\newcommand{\\'+name+'}{'+value.replace('.','{,}')+'}\n'
    provenance=dict(schema='supervisor-privacy-report-sources-v2',meeting_date='2026-10-17',
        evidence_cutoff='2026-10-14',prototype_protocol_date='2026-10-17',
        scope='two sections: current architecture and privacy forward specification / reverse inference proof',
        main_release_changed=False,benchmarks_rescored=False,benchmark_results_in_report=False,
        utility_theorems_in_report=False,illustrative_kernel_example_not_empirical=True,
        numerical_sampler_certified=False,main_release_pins_verified=len(manifest['files_sha256']),
        derived=dict(GeoI_unit_exact=str(unit),GeoI_cap_exact=str(cap),
            Q_all_duration_ratio_upper_exact=str(upper),Q_ratio_outward_decimal=ratio,
            binary_equal_prior_posterior_upper_percent=posterior,
            Q_example_weights=list(map(str,weights)),Q_example_scores=list(scores),
            Q_example_allowance=str(epsilon),Q_example_alpha=str(alpha),Q_example_probabilities=list(map(str,probabilities))),
        files_sha256={name:sha(ROOT/name) for name in SOURCES})
    return constants,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();constants,sources=generated()
    outputs={'privacy_constants.tex':constants,'report_sources.json':json.dumps(sources,ensure_ascii=False,indent=2)+'\n'}
    for name,content in outputs.items():
        path=HERE/name
        if args.check:
            if not path.exists() or path.read_text()!=content:raise ValueError('Regenerate '+name)
        else:path.write_text(content)
    print('PASS: privacy constants, source pins and unchanged main release' if args.check else 'Built privacy constants/provenance; no benchmark or sampling')


if __name__=='__main__':main()
