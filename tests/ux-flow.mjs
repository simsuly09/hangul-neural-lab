import fs from 'node:fs/promises';
const root=new URL('../',import.meta.url);
const names=['index.html','app.mjs','runtime.mjs','tour.mjs','model/network.json'];
const source=Object.fromEntries(await Promise.all(names.map(async name=>[name,await fs.readFile(new URL(name,root),'utf8')])));
const run=async function runUxRegression(s){
 let checks=0;const assert=(v,msg)=>{checks++;if(!v)throw Error(msg);};
 const meta=JSON.parse(s["model/network.json"]);
 const outputs=new Map();
 for(const l of meta.layers){let shape;if(l.type==='InputLayer')shape=[64,64,3];else{const prev=outputs.get(l.inputs[0]).shape,c=l.config;if(l.type==='Conv2D')shape=[0,1].map(i=>c.padding==='same'?Math.ceil(prev[i]/c.strides[i]):Math.floor((prev[i]-c.kernel_size[i])/c.strides[i])+1).concat(c.filters);else if(l.type==='MaxPooling2D')shape=[Math.floor(prev[0]/2),Math.floor(prev[1]/2),prev[2]];else if(l.type==='Flatten')shape=[prev.reduce((a,b)=>a*b,1)];else shape=[c.units];}const data=new Float32Array(shape.reduce((a,b)=>a*b,1));if(l.type==='Dense')data[0]=1;outputs.set(l.name,{shape,data});}
 class El{
  constructor(tag='div'){this.tagName=tag.toUpperCase();this.dataset={};this.hidden=false;this.disabled=false;this.value='';this.children=[];this.attrs={};this.style={};this.listeners={};this.className='';this.open=false;this.clientWidth=320;this.offsetLeft=0;this._text='';this._html='';this.classList={toggle:(name,force)=>{const set=new Set(this.className.split(/\s+/).filter(Boolean));const on=force??!set.has(name);on?set.add(name):set.delete(name);this.className=[...set].join(' ');},contains:n=>this.className.split(/\s+/).includes(n)};}
  set textContent(v){this._text=String(v);this._html='';this.children=[];}get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
  set innerHTML(v){this._html=v;this._text='';this.children=[];}get innerHTML(){return this._html;}
  setAttribute(n,v){this.attrs[n]=String(v);}removeAttribute(n){delete this.attrs[n];}
  append(...xs){xs.forEach(x=>{x.parentElement=this;this.children.push(x);});}appendChild(x){this.append(x);return x;}replaceChildren(...xs){this.children=[];this._text='';this._html='';this.append(...xs);}
  addEventListener(n,f){this.listeners[n]=f;}scrollIntoView(){this.scrolled=true;}scrollTo(){}focus(){document.activeElement=this;}click(){if(!this.disabled)this.onclick?.({target:this});}setPointerCapture(){}getBoundingClientRect(){return {left:0,top:0,width:320,height:320};}
  getContext(){return context;}querySelectorAll(q){return this.children.filter(x=>q==='[role="tab"]'?x.attrs.role==='tab':false);}
 }
 const context={fillRect(){},putImageData(){},beginPath(){},arc(){},fill(){},moveTo(){},lineTo(){},stroke(){},drawImage(){},getImageData(){return {data:new Uint8ClampedArray(320*320*4).fill(255)};},createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};}};
 const elements={};for(const m of s["index.html"].matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"[^>]*>/g)){const el=new El(m[1]);el.hidden=/\shidden(?:\s|>)/.test(m[0]);el.disabled=/\sdisabled(?:\s|>)/.test(m[0]);elements[m[2]]=el;}
 for(const [id,v]of Object.entries({pen:'14',density:'16',speed:'1',timeline:'0'}))elements[id].value=v;
 const makeGroup=(n,key,values)=>Array.from({length:n},(_,i)=>{const el=new El('button');el.dataset[key]=String(values?.[i]??i);return el;});
 const branch=makeGroup(3,'branch'),resultBranch=makeGroup(3,'resultBranch'),journey=makeGroup(3,'journey',[1,2,3]),stages=makeGroup(7,'stage',[0,1,5,9,13,17,18]);
 const tabLists=[branch,resultBranch].map(arr=>{const list=new El();list.children=arr;arr.forEach(el=>el.attrs.role='tab');return list;});
 const resultPanel=new El(),badge=new El(),body=new El();
 const document={body,activeElement:null,createElement:t=>new El(t),querySelector(q){if(q[0]==='#'){if(elements[q.slice(1)])return elements[q.slice(1)];if(elements.operation.innerHTML.includes('id="'+q.slice(1)+'"'))return elements[q.slice(1)]=new El();return null;}if(q==='.result-panel')return resultPanel;if(q==='.result-badge')return badge;if(q==='.layer.active')return elements.layers.children.find(x=>x.classList.contains('active'))||null;throw Error('unknown selector '+q);},querySelectorAll(q){if(q==='[data-journey]')return journey;if(q==='[data-stage]')return stages;if(q==='[data-branch]')return branch;if(q==='[data-result-branch]')return resultBranch;if(q==='[data-branch], [data-result-branch]')return [...branch,...resultBranch];if(q==='[role="tablist"]')return tabLists;if(q==='.layer')return elements.layers.children;throw Error('unknown selectorAll '+q);}};
 let clock=0,resolveInference;
 const network={meta,model:{dispose(){}},weights:new Map(meta.weights.map(w=>[w.name,new Float32Array(w.length)])),infer:async()=>outputs};
 const window={matchMedia:()=>({matches:false})};
 const scene={controls:{enabled:true},hover:{hidden:true},prepare(){},setStage(index){this.index=index;},setProgress(index){this.index=index;},select(){},setInspection(){},reset(){},focusLayer(){},disposeRoot(){}};
 const strip=code=>code.replace(/^import .*;\r?$/gm,'').replace(/^export /gm,'');
 const exports=new Function(strip(s["runtime.mjs"])+'\nreturn {branchLayers,decode,convAt};')();
 const tourExports=new Function('branchLayers',strip(s["tour.mjs"])+'\nreturn {buildTour,InferenceTour};')(exports.branchLayers);
 const deps={...exports,...tourExports,document,window,performance:{now:()=>clock},requestAnimationFrame:cb=>{if(cb.name!=='animate')Promise.resolve().then(()=>cb(clock));},tf:{ready:async()=>{},memory:()=>({numTensors:55})},NetworkScene:class{},readModel:async()=>null,saveModel:async()=>{},forgetModel:async()=>{},loadNetwork:async()=>network};
 const source=strip(s["app.mjs"]).replace(/\nboot\(\);\s*$/,'')+`\nreturn {install(n,v){network=n;scene=v;sampleTensor=new Float32Array(12288).fill(1);},infer,get tour(){return tour;},get epoch(){return inferenceEpoch;},get revealed(){return resultRevealed;},resetPlayback,selectLayer,get branch(){return branch;},activateModel};`;
 const api=new Function(...Object.keys(deps),source)(...Object.values(deps));
 assert(body.dataset.phase==='loading','initial loading');
 assert(!elements.tourCaption.hidden,'initial guide visible');
 assert(elements.resultContent.hidden,'initial result concealed');
 assert(stages.every(x=>x.disabled),'stage buttons initially disabled');
 api.install(network,scene);
 await api.infer();
 assert(body.dataset.phase==='playing','inference starts playback');
 assert(api.tour.stages.length===19&&api.tour.total===34000,'parallel timeline unchanged');
 assert(scene.index===0&&window.playbackState.layers.length===3,'three branches begin together');
 assert(!api.revealed&&elements.resultContent.hidden,'no result at playback start');
 assert(stages.every(x=>!x.disabled),'major stage navigation enabled');
 api.tour.tick(1000);assert(elements.timeLabel.textContent==='0:01 / 0:34','time updates between stage boundaries');
 elements.play.click();assert(body.dataset.phase==='paused'&&!api.tour.playing,'pause state');
 elements.play.click();assert(body.dataset.phase==='playing','resume state');
 const stageNames=[0,1,5,9,13,17,18];
 for(let i=0;i<7;i++){stages[i].click();assert(api.tour.index===stageNames[i],'stage jump '+i);assert(!api.tour.playing&&body.dataset.phase==='paused','stage jump pauses '+i);assert(scene.index===stageNames[i],'views synchronized '+i);assert(elements.resultContent.hidden&&!api.revealed,'jump conceals result '+i);assert(stages[i].classList.contains('active'),'major stage active '+i);}
 const timeBefore=api.tour.time;branch[1].click();assert(api.branch===1&&api.tour.time===timeBefore,'detail branch keeps time');assert(resultBranch[1].classList.contains('active'),'branch tabs synchronized');
 api.tour.seek(.5);elements.nextStep.click();assert(api.tour.index>0,'next step works');elements.previousStep.click();assert(!api.revealed,'previous step keeps concealment');
 api.tour.seek(1);assert(api.revealed&&!elements.resultContent.hidden&&elements.resultPending.hidden,'result revealed at end');assert(elements.result.textContent==='가','decoded result displayed');assert(body.dataset.phase==='complete'&&resultPanel.scrolled,'completion guides to result');
 assert(elements.timeLabel.textContent==='0:34 / 0:34','completed clock');
 assert(elements.probabilities.innerHTML.includes('100.0%'),'real decoded probability display');
 resultBranch[2].click();assert(api.branch===2&&branch[2].classList.contains('active'),'result tabs update detail branch');
 elements.play.click();assert(elements.resultContent.hidden&&!api.revealed&&api.tour.index===0,'replay conceals result');assert(body.dataset.phase==='playing','replay phase');
 api.tour.seek(1);elements.newDrawing.click();assert(!api.tour&&elements.resultContent.hidden,'new drawing resets result');assert(elements.drawing.scrolled&&document.activeElement===elements.drawing,'new drawing focuses input');assert(body.dataset.phase==='ready','new drawing ready state');assert(stages.every(x=>x.disabled),'new drawing disables stage jumps');
 elements.pen.value='21';elements.pen.oninput();assert(elements.penValue.textContent==='21','pen value updates');
 let keyPrevented=false;document.activeElement=branch[0];tabLists[0].listeners.keydown({key:'ArrowRight',preventDefault(){keyPrevented=true;}});assert(keyPrevented&&api.branch===1&&document.activeElement===branch[1],'keyboard tabs');
 api.install(network,scene);network.infer=()=>new Promise(r=>resolveInference=r);const pending=api.infer();await Promise.resolve();await Promise.resolve();assert(body.dataset.phase==='computing','computing state');elements.clear.click();resolveInference(outputs);await pending;assert(!api.tour&&!api.revealed&&elements.resultContent.hidden,'cleared input discards stale inference');
 assert(!elements.predict.disabled&&elements.predict.textContent==='신경망으로 살펴보기','CTA restored after stale inference');
 const beforeReady=body.dataset.phase;await api.activateModel();assert(body.dataset.phase==='ready','model activation ready');
 assert(!/<details[^>]*class="model-loader"[^>]*\sopen/.test(s['index.html']),'model settings closed');
 assert(!/<details[^>]*id="technicalDetails"[^>]*\sopen/.test(s['index.html']),'technical details closed');
 return {checks,stages:19,duration:34000,branches:3,browserVisualReview:false};
};
console.log(JSON.stringify(await run(source),null,2));
