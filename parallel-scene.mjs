import {NetworkScene} from './scene.mjs';
import {branchLayers} from './runtime.mjs';

const names=['초성','중성','종성'];
const titles={InputLayer:'입력 이미지',Conv2D:'합성곱 + ReLU',MaxPooling2D:'최대 풀링',Flatten:'펼치기',Dense:'분류'};
export class ParallelNetworkScene{
 constructor(element,onPick){
  this.element=element;this.views=[0,1,2].map(i=>new NetworkScene(element.querySelector('[data-branch-scene="'+i+'"]'),name=>onPick(name,i),{accent:[0x9bdad1,0x9fc7ff,0xd2baff][i]}));
  const owner=this;
  this.controls={get enabled(){return owner.views.every(v=>v.controls.enabled);},set enabled(value){owner.views.forEach(v=>v.controls.enabled=value);}};
  this.hover={set hidden(value){owner.views.forEach(v=>v.hover.hidden=value);}};
  this.frameIndex=-1;this.outputs=null;this.prepared=null;this.lastScan=-Infinity;
 }
 prepare(network,outputs,density){
  if(this.prepared===network&&this.outputs===outputs&&this.density===density)return;
  this.prepared=network;this.outputs=outputs;this.density=density;this.chains=[0,1,2].map(b=>branchLayers(network.meta,b));
  this.views.forEach((view,i)=>{view.build(this.chains[i],outputs,this.chains[i],density,false,null,{preserveCamera:!!view.outputs});view.installDenseConnections(network.weights);});
  this.frameIndex=-1;
 }
 setStage(index,{duration=900,focus=true}={}){
  if(!this.chains)return;
  this.frameIndex=index;
  this.views.forEach((view,i)=>{
   const layer=this.chains[i][index];view.select(layer.name);
   if(focus)view.focusLayer(layer.name,duration);else view.reset();
   const title=this.element.querySelector('[data-branch-stage="'+i+'"]');
   const text=names[i]+' · '+titles[layer.type]+' · '+(index+1)+' / '+this.chains[i].length;
   if(title.textContent!==text)title.textContent=text;
  });
  this.animateInspection(0,true);
 }
 setProgress(index,local,complete=false){
  if(!this.chains)return;
  this.views.forEach((view,i)=>view.setProgress(complete?1:(index-1+local)/(this.chains[i].length-1)));
  this.animateInspection(local);
 }
 animateInspection(local,force=false){
  const now=performance.now();if(!force&&(this.inspectionPaused||now-this.lastScan<100))return;this.lastScan=now;
  if(!this.chains||this.frameIndex<0)return;
  this.views.forEach((view,i)=>{
   const layer=this.chains[i][this.frameIndex],tensor=this.outputs.get(layer.name);
   if(!['Conv2D','MaxPooling2D'].includes(layer.type))return;
   const [h,w]=tensor.shape,pos=Math.min(h*w-1,Math.floor(local*h*w));
   view.setInspection(layer.name,Math.floor(pos/w),pos%w,0,0,layer.config.kernel_size||layer.config.pool_size,local);
  });
 }
 indexFor(name){for(const chain of this.chains||[]){const i=chain.findIndex(l=>l.name===name);if(i>=0)return i;}return -1;}
 focusLayer(name,duration=900){const index=this.indexFor(name);if(index>=0)this.setStage(index,{duration,focus:true});}
 select(name){const index=this.indexFor(name);if(index>=0){this.frameIndex=index;this.views.forEach((view,i)=>view.select(this.chains[i][index].name));}}
 setInspection(name,...args){if(!this.chains)return;const branch=this.chains.findIndex(c=>c.some(l=>l.name===name));if(branch>=0)this.views[branch].setInspection(name,...args);}
 reset(){this.views.forEach(v=>v.reset());}
 disposeRoot(){
  this.views.forEach(v=>{v.disposeRoot();v.outputs=null;v.cameraMotion=null;v.hit=null;v.hover.hidden=true;});
  this.prepared=null;this.outputs=null;this.frameIndex=-1;
  this.element.querySelectorAll('[data-branch-stage]').forEach(el=>el.textContent='입력 대기');
 }
}
