import assert from 'node:assert/strict';
import {prefixEvents,centroid,localAnswer,qAllowance,qPrefix,posteriorUpper} from '../web/static/thesis_demo/logic.mjs';

// Fake events are retained by clock, not by a fixed number of packets.
assert.deepEqual(prefixEvents([{t:0},{t:9},{t:20},{t:29}],20),[{t:0},{t:9},{t:20}]);
const order=[0,1,2,3,4,5,6,7];
const answer=localAnswer(order,[7,5,3,1,0,4,6]);
assert.deepEqual(answer.answer,[0,1,3,4,5,6,7]); // full list, not top five
assert.equal(answer.recall,.8);
assert.deepEqual(answer.missing,[2]);
assert.equal(localAnswer([],[]).recall,null);
assert.equal(localAnswer(order,[]).recall,0);
assert.deepEqual(centroid([[1,2],[3,4]]),[2,3]);
let spent=0;
for(let j=1;j<=10000;j++){
  spent+=qAllowance(j);
  assert.ok(Math.abs(spent-qPrefix(j))<1e-12);
  assert.ok(spent<1);
}
assert.ok(Math.abs(posteriorUpper(.5)-.709080)<1e-5);
assert.equal(posteriorUpper(0),0);assert.equal(posteriorUpper(1),1);
assert.ok(posteriorUpper(.9)>.95);
console.log('PASS: causal wire, full ranking, N/A, Q composition, Bayes interpretation');
