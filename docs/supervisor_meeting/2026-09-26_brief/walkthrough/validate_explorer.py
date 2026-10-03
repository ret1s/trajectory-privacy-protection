"""Execute the explorer JS against a minimal DOM; not a browser rendering test."""
import json
import re
from pathlib import Path
import quickjs

OUT=Path(__file__).resolve().parent
html=(OUT/'walkthrough.html').read_text()
context=quickjs.Context()
context.eval('''
class Element {
 constructor(){this.children=[];this.style={};this.dataset={};this.value='';this.textContent='';this.listeners={};}
 append(e){this.children.push(e)}
 replaceChildren(){this.children=[]}
 setAttribute(k,v){this[k]=v}
 addEventListener(k,v){this.listeners[k]=v}
 getBoundingClientRect(){return {x:0,y:0}}
}
const nodes={};
const document={
 getElementById(id){return nodes[id]||(nodes[id]=new Element())},
 createElement(){return new Element()},
 createElementNS(){return new Element()},
 querySelectorAll(){return [0,60,80,120,360,384].map(t=>{let e=new Element();e.dataset.t=t;return e})}
};
const innerWidth=1280,innerHeight=800;
document.getElementById('mode').value='boundary';
document.getElementById('time').value=6;
''')
context.eval(re.findall(r'<script>(.*?)</script>',html,re.S)[-1])
result=json.loads(context.eval('''JSON.stringify((()=>{
 let renders=0;
 for(const key of ['plain','boundary'])for(let j=0;j<21;j++)for(let c=0;c<6;c++){
  mode.value=key;slider.value=j;category.value=c;render();
  const row=D.runs[key].rows[j];
  if($('clock').textContent!==row.t+' giây')throw Error('clock');
  if($('known').textContent!==row.service.cached_union_count)throw Error('cache count');
  if($('timeline').children.length!==21)throw Error('timeline');
  if($('returned').children.length!==Math.max(1,row.service.returned[c].length))throw Error('POI rows');
  if(!$('map').children.length)throw Error('map');
  renders++;
 }
 mode.value='boundary';slider.value=6;category.value=5;render();
 if($('published').textContent!=='60s')throw Error('delayed first publication');
 $('prev').onclick();if(+slider.value!==5)throw Error('previous');
 $('next').onclick();if(+slider.value!==6)throw Error('next');
 slider.value=20;render();if(!$('close').textContent.includes('4 bộ chờ bị hủy'))throw Error('close');
 return {status:'passed',renders,browser_rendering_checked:false};
})())'''))
(OUT/'explorer_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
