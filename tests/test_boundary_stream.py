import unittest
from core.boundary_release import BoundaryPolicy, BoundaryProtectedStream

class Stub:
    def reset(self): self.calls=[]
    def protect_step(self,lat,lon,t):
        self.calls.append((t,lat,lon));return ((lat,lon),)

class BoundaryStreamTests(unittest.TestCase):
    def test_head_never_reaches_core_and_tail_never_flushes(self):
        e=Stub();s=BoundaryProtectedStream(e,BoundaryPolicy(40,40));out=[]
        for t in range(0,121,20):out+=s.ingest(t,10+t/1000,20)
        self.assertEqual([x[0] for x in e.calls],[40,60,80,100,120])
        self.assertEqual([x.timestamp_s for x in out],[80,100,120])
        self.assertEqual([x.candidates[0].lat for x in out],[10.04,10.06,10.08])
        self.assertEqual(s.close(120),());self.assertEqual(s.evaluator_summary()['tail_cancelled'],2)
    def test_prefix_causality(self):
        def run(future):
            s=BoundaryProtectedStream(Stub(),BoundaryPolicy(20,20));out=[]
            for t in range(0,81,20):
                got=s.ingest(t,10 if t<=40 else future,20)
                if t<=40:out+=got
            return out
        self.assertEqual(run(11),run(40))
    def test_head_coordinates_cannot_influence_public_output(self):
        def run(secret):
            s=BoundaryProtectedStream(Stub(),BoundaryPolicy(40,20));out=[]
            for t in range(0,101,20):out+=s.ingest(t,secret if t<40 else 10,20)
            return out
        self.assertEqual(run(0),run(80))
    def test_zero_gate_equivalence(self):
        s=BoundaryProtectedStream(Stub(),BoundaryPolicy(0,0))
        for i in range(3):
            o=s.ingest(i,10+i,20);self.assertEqual(o[0].candidates[0].lat,10+i)
        s.close(3);self.assertEqual(s.cancelled,0)
    def test_irregular_schedule_no_early_release(self):
        s=BoundaryProtectedStream(Stub(),BoundaryPolicy(0,60),session_start_s=100)
        self.assertFalse(s.ingest(100,1,1));self.assertFalse(s.ingest(130,2,2))
        out=s.ingest(200,3,3);self.assertEqual(len(out),2)
        self.assertTrue(all(x.timestamp_s==100 for x in out))
    def test_short_session_and_invalid_calls(self):
        s=BoundaryProtectedStream(Stub(),BoundaryPolicy(60,60));self.assertFalse(s.ingest(0,1,1));s.close(20)
        with self.assertRaises(ValueError):s.ingest(21,1,1)
        with self.assertRaises(ValueError):s.close(21)
        for x in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):BoundaryPolicy(x,0)
    def test_public_schema_no_private_timestamps_or_labels(self):
        s=BoundaryProtectedStream(Stub(),BoundaryPolicy(0,20));s.ingest(0,1,1)
        d=s.ingest(20,2,2)[0].to_dict();self.assertEqual(set(d),{'event_id','timestamp_s','candidates'})
        self.assertEqual(d['event_id'],'event_000000');self.assertEqual(d['timestamp_s'],20)
    def test_existing_budget_is_not_reset_per_release(self):
        e=Stub();s=BoundaryProtectedStream(e,BoundaryPolicy(0,20))
        for t in range(40):s.ingest(t,1,1)
        self.assertEqual(len(e.calls),40)
        self.assertEqual(s.generated,40)

if __name__=='__main__':unittest.main()
