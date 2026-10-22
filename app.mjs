import {loadNetwork,branchLayers,decode,convAt} from './runtime.mjs';
import {NetworkScene} from './scene.mjs';
import {readModel,saveModel,forgetModel} from './model-cache.mjs';

const $=s=>document.querySelector(s),canvas=$('#drawing'),ctx=canvas.getContext('2d',{willReadFrequently:true});
const branchNames=['초성','중성','종성'];let network,outputs=null,prediction,branch=0,selected=null,chain=[],history=[],drawing=false,dirty=false,sampleTensor=null,sampleLabel=null;
let scene,focus=false,playing=false,playPosition=0,lastTime=0,inspectPosition=0,inspectionPaused=false;
const labelFor=l=>l.type==='InputLayer'?'입력':l.type==='Conv2D'?'합성곱':l.type==='MaxPooling2D'?'풀링':l.type==='Flatten'?'Flatten':'Softmax';
function clearCanvas(){ctx.fillStyle='#fff';ctx.fillRect(0,0,320,320);dirty=false;sampleTensor=null;sampleLabel=null;$('#drawHint').hidden=false;}
clearCanvas();
function checkpoint(){history.push(ctx.getImageData(0,0,320,320));if(history.length>25)history.shift();}
function invalidate(){dirty=true;sampleTensor=null;sampleLabel=null;$('#drawHint').hidden=true;if(outputs){$('#message').textContent='입력이 바뀌었습니다. 다시 인식하면 결과와 활성값을 갱신합니다.';}}
function point(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*320/r.width,(e.clientY-r.top)*320/r.height];}
canvas.addEventListener('pointerdown',e=>{checkpoint();drawing=true;canvas.setPointerCapture(e.pointerId);const [x,y]=point(e);ctx.strokeStyle='#000';ctx.fillStyle='#000';ctx.lineWidth=+$('#pen').value;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();ctx.arc(x,y,ctx.lineWidth/2,0,2*Math.PI);ctx.fill();ctx.beginPath();ctx.moveTo(x,y);invalidate();});
canvas.addEventListener('pointermove',e=>{if(!drawing)return;ctx.lineTo(...point(e));ctx.stroke();});
canvas.addEventListener('pointerup',()=>{drawing=false;});canvas.addEventListener('pointercancel',()=>drawing=false);
$('#clear').onclick=()=>{checkpoint();clearCanvas();$('#message').textContent='한글 한 글자를 쓰고 인식하기를 눌러 주세요.';};
$('#undo').onclick=()=>{const image=history.pop();if(image){ctx.putImageData(image,0,0);invalidate();}};
$('#aboutButton').onclick=()=>$('#about').showModal();$('#closeAbout').onclick=()=>$('#about').close();

function normalizedInput(){
  const image=ctx.getImageData(0,0,320,320);let x0=320,x1=-1,y0=320,y1=-1,count=0;
  for(let y=0;y<320;y++)for(let x=0;x<320;x++){const i=(y*320+x)*4;if((image.data[i]+image.data[i+1]+image.data[i+2])/3<220){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);count++;}}
  if(count<16)throw Error('그림판에 한글 한 글자를 써 주세요.');
  const out=document.createElement('canvas');out.width=64;out.height=64;const c=out.getContext('2d',{willReadFrequently:true});c.fillStyle='#fff';c.fillRect(0,0,64,64);c.imageSmoothingEnabled=true;c.imageSmoothingQuality='high';const w=x1-x0+1,h=y1-y0+1,scale=56/Math.max(w,h);c.drawImage(canvas,x0,y0,w,h,(64-w*scale)/2,(64-h*scale)/2,w*scale,h*scale);
  const rgba=c.getImageData(0,0,64,64);$('#preview').getContext('2d').putImageData(rgba,0,0);const pixels=new Float32Array(64*64*3);for(let i=0,j=0;i<rgba.data.length;i+=4){pixels[j++]=rgba.data[i]/255;pixels[j++]=rgba.data[i+1]/255;pixels[j++]=rgba.data[i+2]/255;}return pixels;
}
function previewPixels(pixels){const c=$('#preview').getContext('2d');const im=c.createImageData(64,64);for(let i=0;i<4096;i++){im.data[i*4]=pixels[i*3]*255;im.data[i*4+1]=pixels[i*3+1]*255;im.data[i*4+2]=pixels[i*3+2]*255;im.data[i*4+3]=255;}c.putImageData(im,0,0);}
function overview(){const pools=chain.filter(l=>l.type==='MaxPooling2D');return [chain[0],chain.find(l=>l.type==='Conv2D'),...pools,...chain.filter(l=>['Flatten','Dense'].includes(l.type))];}
function refreshScene(){if(!outputs)return;scene?.build(overview(),outputs,chain,+$('#density').value,focus,selected);scene?.setProgress(playPosition);const layer=network.meta.layers.find(l=>l.name===selected);const t=outputs.get(selected);$('#sampling').textContent=`전체 ${chain.length}개 층을 계산합니다. 개요: 주요 8개 층 · 각 층 최대 4개 채널 · 공간 최대 ${$('#density').value} × ${$('#density').value} 위치 · Flatten 최대 192개 값. 밝기는 층별로 조절됩니다.`;}
function selectLayer(name){selected=name;document.querySelectorAll('.layer').forEach(el=>el.classList.toggle('active',el.dataset.name===name));scene?.select(name);if(focus)refreshScene();inspectPosition=0;renderOperation();}
function renderLayers(){chain=branchLayers(network.meta,branch);if(!chain.some(l=>l.name===selected))selected=chain.find(l=>l.type==='Conv2D').name;const holder=$('#layers');holder.replaceChildren();chain.forEach((l,i)=>{const button=document.createElement('button');button.className='layer'+(selected===l.name?' active':'');button.dataset.name=l.name;const shape=outputs?.get(l.name)?.shape;button.textContent=String(i).padStart(2,'0')+' '+labelFor(l);const span=document.createElement('span');span.textContent=shape?shape.join('×'):l.name;button.appendChild(span);button.onclick=()=>selectLayer(l.name);holder.appendChild(button);});}
function renderResult(){const {meta}=network;$('#result').textContent=prediction.char;$('#resultMeta').textContent=sampleLabel?`검증 손글씨 정답: ${sampleLabel} · ${prediction.char===sampleLabel?'일치':'불일치'}`:'손글씨 입력의 추론 결과';$('#parts').innerHTML=prediction.ranked.map((r,i)=>`<div><small>${branchNames[i]}</small><strong>${meta.labels[i][r[0].i]}</strong></div>`).join('');$('#probabilities').innerHTML=prediction.ranked[branch].slice(0,5).map(v=>`<div class="prob-row"><span>${meta.labels[branch][v.i]}</span><div class="bar"><i style="width:${v.p*100}%"></i></div><span>${(v.p*100).toFixed(1)}%</span></div>`).join('');$('#candidates').innerHTML=prediction.candidates.map(c=>`<span class="candidate" title="세 확률의 곱: ${c.score.toPrecision(5)}">${c.char}</span>`).join('');}
async function infer(){if(!network)return;const btn=$('#predict');btn.disabled=true;btn.textContent='신경망 계산 중…';try{const pixels=sampleTensor||normalizedInput();previewPixels(pixels);$('#message').textContent='전체 신경망을 계산하고 있습니다.';await new Promise(r=>requestAnimationFrame(r));const start=performance.now();outputs=await network.infer(pixels);const elapsed=performance.now()-start;prediction=decode(network.meta,outputs);$('#latency').textContent=elapsed<1000?elapsed.toFixed(0)+' ms':(elapsed/1000).toFixed(2)+' s';renderLayers();renderResult();$('#sceneGuide').hidden=true;playPosition=0;playing=true;lastTime=performance.now();$('#timeline').disabled=false;$('#play').disabled=false;$('#focus').disabled=false;$('#play').textContent='일시정지';refreshScene();renderOperation();$('#message').textContent='추론 완료. 재생은 계산 결과를 천천히 보여줍니다.';window.lastInference={char:prediction.char,expected:sampleLabel,ms:elapsed,outputs:network.meta.outputs.map(n=>Array.from(outputs.get(n).data)),finite:[...outputs.values()].every(v=>v.data.every(Number.isFinite)),tensors:tf.memory().numTensors};}catch(e){$('#message').textContent=e.message;console.error(e);}finally{btn.disabled=false;btn.textContent='손글씨 인식하기';}}
$('#predict').onclick=infer;
document.querySelectorAll('[data-branch]').forEach(button=>button.onclick=()=>{branch=+button.dataset.branch;document.querySelectorAll('[data-branch]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-selected',b===button?'true':'false');});if(network){selected=null;renderLayers();if(outputs){renderResult();refreshScene();renderOperation();}}});
$('#resetCamera').onclick=()=>scene?.reset();$('#density').onchange=refreshScene;$('#focus').onclick=()=>{focus=!focus;$('#focus').textContent=focus?'전체 흐름 보기':'선택 층 확대';refreshScene();};
$('#play').onclick=()=>{if(playPosition>=1)playPosition=0;playing=!playing;lastTime=performance.now();$('#play').textContent=playing?'일시정지':'재생';};
$('#timeline').oninput=()=>{playPosition=+$('#timeline').value/1000;playing=false;$('#play').textContent='재생';scene?.setProgress(playPosition);};
function matrixHTML(values,k){return `<div class="matrix" style="--k:${k}">${values.map(v=>`<span>${v.toFixed(3)}</span>`).join('')}</div>`;}
function renderOperation(){if(!outputs)return;const layer=network.meta.layers.find(l=>l.name===selected),tensor=outputs.get(selected),op=$('#operation');op.className='';$('#operationTitle').textContent=labelFor(layer)+' · '+layer.name;
  if(layer.type==='Conv2D'){
    $('#operationDesc').textContent='입력 한 영역과 학습된 필터를 곱합니다. 모든 입력 채널의 곱을 합하고 편향과 ReLU를 적용합니다.';
    const input=outputs.get(layer.inputs[0]);inspectPosition=Math.floor(tensor.shape[0]/2)*tensor.shape[1]+Math.floor(tensor.shape[1]/4);op.innerHTML=`<div class="calc-controls"><label>출력 채널 <input id="outChannel" aria-label="출력 채널" type="number" min="0" max="${tensor.shape[2]-1}" value="0"></label><label>입력 채널 <input id="inChannel" aria-label="입력 채널" type="number" min="0" max="${input.shape[2]-1}" value="0"></label><button id="scan" class="quiet">${inspectionPaused?'필터 이동 재생':'필터 이동 멈춤'}</button><span id="scanPos"></span></div><div class="calc-row"><div><div class="matrix-label">입력 영역</div><div id="inputMatrix"></div></div><span class="operator">×</span><div><div class="matrix-label">학습된 필터</div><div id="filterMatrix"></div></div><div class="calc-result" id="calcResult"></div></div>`;
    $('#scan').onclick=()=>{inspectionPaused=!inspectionPaused;$('#scan').textContent=inspectionPaused?'필터 이동 재생':'필터 이동 멈춤';};$('#outChannel').oninput=updateConv;$('#inChannel').oninput=updateConv;updateConv();
  }else if(layer.type==='MaxPooling2D'){
    const source=outputs.get(layer.inputs[0]),[h,w,c]=source.shape;const patch=[source.data[0],source.data[c],source.data[w*c],source.data[(w+1)*c]];
    $('#operationDesc').textContent='각 채널의 2 × 2 영역에서 가장 큰 값을 남깁니다. 공간 크기가 절반으로 줄어듭니다.';op.innerHTML=`<div class="calc-row">${matrixHTML(patch,2)}<span class="operator">→</span><div class="calc-result">채널 0, 위치 (0, 0)<br>최댓값 <strong>${Math.max(...patch).toFixed(6)}</strong><br>실제 출력 <strong>${tensor.data[0].toFixed(6)}</strong></div></div>`;
  }else if(layer.type==='Flatten'){
    const prev=outputs.get(layer.inputs[0]);$('#operationDesc').textContent='높이·너비·채널 순서의 배열을 1차원으로 펼칩니다. 값과 순서는 그대로 유지됩니다.';op.innerHTML=`<div class="calc-result">${prev.shape.join(' × ')} → <strong>${tensor.data.length}</strong>개 값<br>3D 재생에서 일부 값이 공간 격자에서 벡터로 이동합니다.<br>앞 8개 값: <span class="number">${Array.from(tensor.data.slice(0,8)).map(x=>x.toFixed(3)).join(', ')}</span></div>`;
  }else if(layer.type==='Dense'){
    const a=outputs.get(layer.inputs[0]).data;const wi=network.meta.weights.find(w=>w.layer===layer.name&&w.name.includes('kernel'));const weights=network.weights.get(wi.name);const bias=network.weights.get(layer.name+'/bias:0');const rank=prediction.ranked[branch][0];let sum=bias[rank.i];for(let i=0;i<a.length;i++)sum+=a[i]*weights[i*layer.config.units+rank.i];$('#operationDesc').textContent='펼친 값과 학습된 가중치의 곱을 모두 더하고, Softmax로 각 후보의 출력 확률을 계산합니다.';op.innerHTML=`<div class="calc-result">선택된 후보 <strong>${network.meta.labels[branch][rank.i]}</strong><br>${a.length}개 값의 가중합 + 편향 = <strong>${sum.toFixed(6)}</strong><br>Softmax 출력 = <strong>${rank.p.toFixed(6)}</strong> (${(rank.p*100).toFixed(2)}%)</div><div class="dense-bars">${Array.from(tensor.data).map(v=>`<i style="height:${Math.max(2,v*100)}%"></i>`).join('')}</div>`;
  }else{$('#operationDesc').textContent='모델에 실제로 전달한 64 × 64 RGB 입력입니다. 흰색은 1, 검은색은 0입니다.';op.innerHTML=`<div class="calc-result">입력 값 수 <strong>${tensor.data.length}</strong><br>최솟값 ${Math.min(...tensor.data).toFixed(3)} · 최댓값 ${Math.max(...tensor.data).toFixed(3)}</div>`;}
}
function updateConv(){if(!outputs||!$('#outChannel'))return;const t=outputs.get(selected),[h,w,c]=t.shape;const layer=network.meta.layers.find(l=>l.name===selected),cin=outputs.get(layer.inputs[0]).shape[2];const channel=Math.max(0,Math.min(c-1,+$('#outChannel').value||0)),inputChannel=Math.max(0,Math.min(cin-1,+$('#inChannel').value||0));const pos=inspectPosition%(h*w),y=Math.floor(pos/w),x=pos%w;const result=convAt(network,outputs,selected,y,x,channel,inputChannel);$('#scanPos').textContent=`위치 (${y}, ${x})`;$('#inputMatrix').innerHTML=matrixHTML(result.patch,result.kw);$('#filterMatrix').innerHTML=matrixHTML(result.filter,result.kw);$('#calcResult').innerHTML=`${result.cin}개 입력 채널 합산<br>편향 <strong>${result.bias.toFixed(5)}</strong><br>가중합 <strong>${result.sum.toFixed(5)}</strong><br>ReLU 출력 <strong>${result.actual.toFixed(5)}</strong>`;scene?.setInspection(selected,y,x,channel,inputChannel,layer.config.kernel_size);}
let scanClock=0;function animate(now){requestAnimationFrame(animate);const dt=Math.min(100,now-lastTime);lastTime=now;if(playing&&outputs){playPosition=Math.min(1,playPosition+dt/16000*+$('#speed').value);scene?.setProgress(playPosition);$('#timeline').value=Math.round(playPosition*1000);const step=Math.min(chain.length-1,Math.floor(playPosition*(chain.length-1)));$('#timeLabel').textContent=`${step+1} / ${chain.length} 단계`;if(playPosition===1){playing=false;$('#play').textContent='다시 재생';}}if(outputs&&!inspectionPaused&&now-scanClock>250){inspectPosition++;scanClock=now;updateConv();}}
requestAnimationFrame(animate);
const basicStrokes={
  '가':[[[65,90],[145,90],[145,235]],[[220,65],[220,255]],[[220,155],[280,155]]],
  '나':[[[65,85],[65,235],[145,235]],[[220,65],[220,255]],[[220,155],[280,155]]],
  '다':[[[145,80],[65,80],[65,235],[145,235]],[[220,65],[220,255]],[[220,155],[280,155]]]
};
function drawBasic(target,label){const c=target.getContext('2d');c.fillStyle='#fff';c.fillRect(0,0,target.width,target.height);c.strokeStyle='#111';c.lineWidth=15*target.width/320;c.lineCap='round';c.lineJoin='round';for(const stroke of basicStrokes[label]){c.beginPath();stroke.forEach(([x,y],i)=>{const p=[x*target.width/320,y*target.height/320];i?c.lineTo(...p):c.moveTo(...p);});c.stroke();}}
async function useBasic(label){checkpoint();clearCanvas();drawBasic(canvas,label);invalidate();if(network)await infer();else $('#message').textContent='예시를 그렸습니다. 모델 파일을 불러오면 인식할 수 있습니다.';}
$('#upload').onchange=async e=>{const file=e.target.files[0];if(!file)return;const url=URL.createObjectURL(file);try{const image=new Image();image.src=url;await image.decode();checkpoint();clearCanvas();const s=300/Math.max(image.width,image.height);ctx.drawImage(image,(320-image.width*s)/2,(320-image.height*s)/2,image.width*s,image.height*s);invalidate();$('#message').textContent='이미지를 불러왔습니다. 손글씨 인식하기를 눌러 주세요.';}catch{$('#message').textContent='이미지를 읽을 수 없습니다.';}finally{URL.revokeObjectURL(url);}};
async function activateModel(local=null){
  const previous=network;
  $('#predict').disabled=true;$('#modelFiles').disabled=true;
  try{
    await tf.ready();const next=await loadNetwork(tf,'./',p=>$('#status').textContent=`모델 준비 ${Math.round(p*100)}%`,local);
    network=next;previous?.model.dispose();outputs=null;prediction=null;selected=null;playing=false;playPosition=0;
    $('#params').textContent=network.meta.parameterCount.toLocaleString();$('#status').textContent='모델 준비 완료';$('#statusDot').style.background='#a8ffc9';$('#predict').textContent='손글씨 인식하기';$('#message').textContent='모델을 불러왔습니다. 한글을 쓰거나 기본 예시를 선택해 주세요.';renderLayers();window.appReady=true;
    $('#play').disabled=true;$('#timeline').disabled=true;$('#focus').disabled=true;
  }finally{$('#modelFiles').disabled=false;$('#predict').disabled=!network;}
}
$('#modelFiles').onchange=async event=>{
  const files=Array.from(event.target.files||[]);const config=files.find(f=>f.name==='network.json'),weights=files.find(f=>f.name==='weights.bin');
  try{if(!config||!weights)throw Error('network.json과 weights.bin 두 파일을 동시에 선택해 주세요.');
    const local={meta:JSON.parse(await config.text()),buffer:await weights.arrayBuffer()};await activateModel(local);
    if($('#rememberModel').checked){try{await saveModel(local);$('#message').textContent+=' 다음 접속에서도 이 브라우저가 기억합니다.';}catch{$('#message').textContent+=' 브라우저 저장 공간을 사용할 수 없어 다음 접속 때 다시 선택해야 합니다.';}}
    else await forgetModel().catch(()=>{});
  }catch(error){$('#status').textContent=network?'기존 모델 사용 중':'모델 선택 대기';$('#message').textContent=error.message;}
  finally{event.target.value='';}
};
$('#forgetModel').onclick=async()=>{try{await forgetModel();$('#message').textContent='기억한 모델을 지웠습니다. 현재 열린 모델은 페이지를 닫을 때까지 사용할 수 있습니다.';}catch{$('#message').textContent='저장 공간을 지우지 못했습니다. 브라우저의 사이트 데이터 설정에서 삭제할 수 있습니다.';}};
async function boot(){try{
    try{scene=new NetworkScene($('#scene'),selectLayer);}catch(e){$('#sceneGuide').innerHTML='<p>이 브라우저에서 3D 그래픽을 사용할 수 없습니다.<br>하드웨어 가속을 지원하는 브라우저에서 열어 주세요.</p>';console.error(e);}
    for(const label of Object.keys(basicStrokes)){const button=document.createElement('button');button.className='sample';button.setAttribute('aria-label',`기본 예시 ${label}`);const drawing=document.createElement('canvas');drawing.width=64;drawing.height=64;drawBasic(drawing,label);const span=document.createElement('span');span.textContent=label;button.append(drawing,span);button.onclick=()=>useBasic(label);$('#samples').appendChild(button);}
    $('#status').textContent='기본 모델 다운로드 중';let cached=null;try{cached=await readModel();}catch{}
    if(cached){try{await activateModel(cached);}catch{await forgetModel().catch(()=>{});await activateModel();}}else await activateModel();
  }catch(e){$('#status').textContent='모델 준비 실패';$('#message').textContent=e.message;console.error(e);}}
boot();
