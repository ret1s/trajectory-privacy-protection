// Pure presentation calculations. No training, sampling, or future observations.
export function prefixEvents(events,t){return events.filter(e=>e.t<=t);}
export function centroid(points){return points.length?[0,1].map(j=>points.reduce((s,p)=>s+p[j],0)/points.length):null;}
export function localAnswer(order,available){
  const received=new Set(available),reference=order.slice(0,5);
  const answer=order.filter(i=>received.has(i));
  const hits=reference.filter(i=>received.has(i));
  return {answer,reference,hits,missing:reference.filter(i=>!received.has(i)),
    recall:reference.length?hits.length/reference.length:null};
}
export function qAllowance(j,gamma=1){return gamma/(j*(j+1));}
export function qPrefix(j,gamma=1){return gamma*j/(j+1);}
export function posteriorUpper(prior,ratio=2.437372){return ratio*prior/(1-prior+ratio*prior);}
export function meters(a,b){return Math.hypot(a[0]-b[0],a[1]-b[1]);}
