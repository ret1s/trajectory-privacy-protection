import pytest
from benchmark.versioned_static_poi_cache import VersionedStaticPoiCache


def build():
    return VersionedStaticPoiCache(['a', 'b'], catalogue_version='map-v1',
        epoch_id='public-eight-trip', start_s=0., end_s=12000.)


def record(name):
    return {'id': name, 'category': 'cafe', 'lat': 39.9, 'lon': 116.}


def scope():
    return {'catalogue_version': 'map-v1', 'epoch_id': 'public-eight-trip'}


def test_static_metadata_persists_across_public_sessions_but_not_availability():
    cache = build()
    cache.receive(0., [record('a')], **scope())
    cache.receive(1500., [record('b')], **scope())
    assert cache.static_ids(11100., **scope()) == ('a', 'b')
    assert cache.storage_limit == 2
    absent = cache.dynamic_candidates(11100., **scope(), status_epoch=0,
        current_known_ids=['a'], current_available_ids=['a'])
    assert absent['eligible'] == () and absent['unknown'] == ('a', 'b')
    current = cache.dynamic_candidates(11100., **scope(), status_epoch=185,
        current_known_ids=['b'], current_available_ids=['b'])
    assert current['eligible'] == ('b',) and current['unknown'] == ('a',)


@pytest.mark.parametrize('mismatch', [{'catalogue_version': 'map-v2'}, {'epoch_id': 'new-public-epoch'}])
def test_public_version_or_epoch_change_invalidates_all_old_metadata(mismatch):
    cache = build()
    cache.receive(0., [record('a')], **scope())
    with pytest.raises(ValueError):
        cache.static_ids(20., **{**scope(), **mismatch})
    with pytest.raises(ValueError):  # A superseded scope cannot be revived.
        cache.static_ids(20., **scope())


def test_epoch_expiry_no_unreceived_pois_and_static_only_payload():
    cache = build()
    cache.receive(0., [record('a')], **scope())
    with pytest.raises(ValueError):
        cache.receive(20., [record('c')], **scope())
    with pytest.raises(ValueError):
        cache.receive(20., [{**record('a'), 'available': True}], **scope())
    assert cache.static_ids(20., **scope()) == ('a',)
    with pytest.raises(ValueError):
        cache.static_ids(12000., **scope())
    with pytest.raises(ValueError):
        cache.static_ids(11999., **scope())


def test_metadata_mutation_and_private_clock_rewind_cannot_reuse_received_record():
    cache = build()
    row = record('a')
    cache.receive(0., [row], **scope())
    row['lon'] = 115.
    assert cache.static_records(600., **scope())[0]['lon'] == 116.
    with pytest.raises(ValueError):
        cache.receive(20., [record('b')], **scope())
