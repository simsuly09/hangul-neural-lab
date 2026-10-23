import {branchLayers} from './runtime.mjs';

const branchNames=['초성','중성','종성'];
const info={
 InputLayer:['입력 이미지','64 × 64 RGB 이미지의 픽셀을 신경망에 전달합니다.',1800],
 Conv2D:['합성곱 + ReLU','학습된 필터로 특징을 찾고 음수 출력을 0으로 바꿉니다.',1600],
 MaxPooling2D:['최대 풀링','각 2 × 2 영역에서 가장 큰 값을 남겨 공간 크기를 줄입니다.',2000],
 Flatten:['펼치기','특징 배열의 순서를 유지하면서 한 줄의 벡터로 펼칩니다.',2600],
 Dense:['분류','특징의 가중합과 Softmax로 후보별 출력을 계산합니다.',2400]
};
export function buildTour(meta){
 const stages=[];
 for(let branch=0;branch<3;branch++){
  const chain=branchLayers(meta,branch);
  chain.forEach((layer,layerIndex)=>{
   if(branch>0&&layer.type==='InputLayer')return;
   const [title,description,duration]=info[layer.type];
   stages.push({branch,layer:layer.name,layerIndex,title,description,duration,
    branchName:branchNames[branch],layerCount:chain.length});
  });
 }
 return stages;
}
export class InferenceTour{
 constructor(stages,onFrame){
  this.stages=stages;this.onFrame=onFrame;this.total=stages.reduce((n,s)=>n+s.duration,0);
  this.time=0;this.playing=false;this.index=-1;this.complete=false;
 }
 frame(){
  let start=0,index=this.stages.length-1;
  for(let i=0;i<this.stages.length;i++){if(this.time<start+this.stages[i].duration-1e-7){index=i;break;}start+=this.stages[i].duration;}
  if(this.time>=this.total)start=this.total-this.stages[index].duration;
  const changed=index!==this.index;this.index=index;this.complete=this.time>=this.total;
  this.onFrame({stage:this.stages[index],index,changed,complete:this.complete,
   local:Math.max(0,Math.min(1,(this.time-start)/this.stages[index].duration)),
   progress:this.time/this.total,playing:this.playing,count:this.stages.length});
 }
 start(){this.time=0;this.index=-1;this.playing=true;this.frame();}
 tick(delta,speed=1){if(!this.playing)return;this.time=Math.min(this.total,this.time+Math.max(0,delta)*speed);if(this.time===this.total)this.playing=false;this.frame();}
 pause(){this.playing=false;this.frame();}
 resume(){if(this.complete){this.time=0;this.index=-1;}this.playing=true;this.frame();}
 seek(progress){this.playing=false;this.time=Math.max(0,Math.min(1,progress))*this.total;this.frame();}
 move(delta){const next=this.index+delta;this.playing=false;if(next>=this.stages.length)this.time=this.total;else this.time=this.stages.slice(0,Math.max(0,next)).reduce((n,s)=>n+s.duration,0);this.frame();}
}
