import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

export class NetworkScene {
  constructor(element,onPick) {
    this.element=element;this.onPick=onPick;this.groups=[];this.meshes=[];this.outputs=null;
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.setClearColor(0x0d121a,1);element.appendChild(this.renderer.domElement);
    this.scene=new THREE.Scene();this.camera=new THREE.PerspectiveCamera(40,1,.1,2000);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.enableDamping=true;this.controls.dampingFactor=.08;this.controls.maxDistance=900;this.controls.minDistance=20;
    this.root=new THREE.Group();this.scene.add(this.root);
    const grid=new THREE.GridHelper(500,40,0x1e3047,0x172334);grid.position.y=-24;grid.material.transparent=true;grid.material.opacity=.45;this.scene.add(grid);
    this.raycaster=new THREE.Raycaster();this.pointer=new THREE.Vector2();this.hover=document.querySelector('#hover');
    this.renderer.domElement.addEventListener('pointermove',e=>this.pointerMove(e));
    this.renderer.domElement.addEventListener('pointerleave',()=>this.hover.hidden=true);
    this.renderer.domElement.addEventListener('click',()=>{if(this.hit)this.onPick(this.hit.layer);});
    new ResizeObserver(()=>this.resize()).observe(element);this.resize();this.reset();
    this.running=true;this.tick();
  }
  resize(){const w=this.element.clientWidth,h=this.element.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
  reset(){const width=this.span||230;this.camera.position.set(width*.26,width*.42,width*.95);this.controls.target.set(0,0,0);this.controls.update();}
  disposeRoot(){if(this.scanner){this.scene.remove(this.scanner);this.scanner.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});this.scanner=null;}for(const g of [...this.root.children]){g.traverse(o=>{o.geometry?.dispose();if(o.material){o.material.map?.dispose();o.material.dispose();}});this.root.remove(g);}this.groups=[];this.meshes=[];}
  label(text,subtitle){const canvas=document.createElement('canvas');canvas.width=512;canvas.height=150;const c=canvas.getContext('2d');c.textAlign='center';c.fillStyle='#b3c8e3';c.font='26px "Malgun Gothic",sans-serif';c.fillText(text,256,55);c.fillStyle='#647f9e';c.font='22px Consolas,monospace';c.fillText(subtitle,256,98);const texture=new THREE.CanvasTexture(canvas);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false}));sprite.scale.set(29,8.5,1);sprite.position.set(0,-17,0);return sprite;}
  build(layers,outputs,chain,density=16,focus=false,selected=null){
    this.disposeRoot();this.outputs=outputs;this.chain=chain;this.progress=1;
    let visible=layers;
    if(focus&&selected){const at=chain.findIndex(l=>l.name===selected);visible=chain.slice(Math.max(0,at-1),at+1);}
    this.span=Math.max(90,(visible.length-1)*33+42);
    visible.forEach((layer,k)=>{
      const tensor=outputs.get(layer.name);if(!tensor)return;
      const shape=tensor.shape,group=new THREE.Group();group.position.x=(k-(visible.length-1)/2)*33;group.userData={layer,index:chain.findIndex(l=>l.name===layer.name)};
      this.root.add(group);this.groups.push(group);
      const data=tensor.data,points=[];const max=Math.max(1e-8,data.reduce((m,v)=>Math.max(m,Math.abs(v)),0));
      let subtitle;
      if(shape.length===3){const [h,w,c]=shape,step=Math.max(1,Math.ceil(Math.max(h,w)/density));const channels=Array.from({length:Math.min(c,4)},(_,i)=>Math.round(i*(c-1)/Math.max(1,Math.min(c,4)-1)));
        for(let ci=0;ci<channels.length;ci++)for(let y=0;y<h;y+=step)for(let x=0;x<w;x+=step){const ch=channels[ci],index=(y*w+x)*c+ch;points.push({x:(x/w-.5)*19,y:(.5-y/h)*19,z:(ci-(channels.length-1)/2)*3.2,index,value:data[index],coords:[y,x,ch]});}
        subtitle=shape.join(' × ');group.userData.sampling={channels,step};
      }else{const n=data.length,limit=layer.type==='Dense'?n:192;for(let j=0;j<Math.min(n,limit);j++){const index=Math.floor(j*n/Math.min(n,limit));const cols=layer.type==='Dense'?1:32;points.push({x:layer.type==='Dense'?0:(j%cols/cols-.5)*22,y:layer.type==='Dense'?((n-1)/2-j)*.8:3-Math.floor(j/cols)*1.4,z:0,index,value:data[index],coords:[index]});}subtitle=String(n)+' values';}
      const geometry=new THREE.BoxGeometry(1,1,.7);const material=new THREE.MeshBasicMaterial({transparent:true,opacity:1});const mesh=new THREE.InstancedMesh(geometry,material,points.length);
      const matrix=new THREE.Matrix4(),color=new THREE.Color();const size=shape.length===3?Math.min(.95,17/density):.55;
      points.forEach((p,i)=>{matrix.compose(new THREE.Vector3(p.x,p.y,p.z),new THREE.Quaternion(),new THREE.Vector3(size,size,size));mesh.setMatrixAt(i,matrix);const v=Math.sqrt(Math.max(0,p.value)/max);color.setRGB(.07+.57*v,.13+.84*v,.22+.57*v);if(layer.type==='InputLayer'){const q=Math.max(0,p.value);color.setRGB(q*.8+.08,q*.85+.1,q*.92+.12);}mesh.setColorAt(i,color);});
      mesh.instanceMatrix.needsUpdate=true;mesh.userData={points,layer,baseX:group.position.x};group.add(mesh);this.meshes.push(mesh);
      const title=layer.type==='InputLayer'?'입력':layer.type==='Conv2D'?'합성곱 + ReLU':layer.type==='MaxPooling2D'?'최대 풀링':layer.type==='Flatten'?'펼치기':'출력 확률';group.add(this.label(title,subtitle));
      if(k){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-22,0,0),new THREE.Vector3(-12,0,0)]),new THREE.LineBasicMaterial({color:0x35506e,transparent:true,opacity:.6}));group.add(line);}
      group.userData.mesh=mesh;group.userData.points=points;
    });
    this.select(selected);this.reset();
  }
  select(name){this.selected=name;if(this.scanner){this.scene.remove(this.scanner);this.scanner.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});this.scanner=null;}for(const group of this.groups){const active=group.userData.layer.name===name;group.userData.mesh.material.opacity=active?1:.86;}}
  setInspection(name,y,x,channel,inputChannel,kernel){
    if(this.scanner){this.scene.remove(this.scanner);this.scanner.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});this.scanner=null;}
    const layer=this.chain?.find(l=>l.name===name);if(!layer)return;
    const source=this.groups.find(g=>g.userData.layer.name===layer.inputs[0]);
    const dest=this.groups.find(g=>g.userData.layer.name===name);if(!source||!dest)return;
    const input=this.outputs.get(layer.inputs[0]),output=this.outputs.get(name);const [h,w,c]=input.shape;
    const group=new THREE.Group();this.scene.add(group);this.scanner=group;
    const inputChannels=source.userData.sampling?.channels||[0];const nearest=inputChannels.reduce((best,v)=>Math.abs(v-inputChannel)<Math.abs(best-inputChannel)?v:best,inputChannels[0]);
    // Wireframe follows an actual receptive field. Hidden channels are not invented.
    if(nearest===inputChannel){const zi=inputChannels.indexOf(inputChannel);const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(kernel[1]/w*19,kernel[0]/h*19,.95)),new THREE.LineBasicMaterial({color:0xf0cf85}));box.position.set(source.position.x+(x/w-.5)*19,(.5-y/h)*19,(zi-(inputChannels.length-1)/2)*3.2);group.add(box);}
    const destChannels=dest.userData.sampling?.channels||[0];if(destChannels.includes(channel)){const marker=new THREE.Mesh(new THREE.SphereGeometry(.38,8,6),new THREE.MeshBasicMaterial({color:0xf1d58c}));marker.position.set(dest.position.x+(x/output.shape[1]-.5)*19,(.5-y/output.shape[0])*19,(destChannels.indexOf(channel)-(destChannels.length-1)/2)*3.2+.8);group.add(marker);}
  }
  setProgress(p){this.progress=p;for(const group of this.groups){const index=group.userData.index;const local=Math.max(0,Math.min(1,p*(this.chain.length-1)-index+1));group.userData.mesh.material.opacity=.08+.92*local;
      if(group.userData.layer.type==='Flatten'){const mesh=group.userData.mesh,points=group.userData.points;const matrix=new THREE.Matrix4();points.forEach((q,i)=>{const c=q.index%256,x=Math.floor(q.index/256)%4,y=Math.floor(q.index/(256*4));const old=new THREE.Vector3(-33+(x/4-.5)*19,(.5-y/4)*19,(c/255-.5)*10);const end=new THREE.Vector3(q.x,q.y,q.z);old.lerp(end,local);matrix.compose(old,new THREE.Quaternion(),new THREE.Vector3(.55,.55,.55));mesh.setMatrixAt(i,matrix);});mesh.instanceMatrix.needsUpdate=true;}
    }}
  pointerMove(event){if(!this.outputs)return;const rect=this.renderer.domElement.getBoundingClientRect();this.pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);this.raycaster.setFromCamera(this.pointer,this.camera);const hit=this.raycaster.intersectObjects(this.meshes)[0];this.hit=null;if(!hit){this.hover.hidden=true;return;}const p=hit.object.userData.points[hit.instanceId],layer=hit.object.userData.layer;this.hit={layer:layer.name,point:p};this.hover.textContent=`${layer.name} [${p.coords.join(', ')}] = ${p.value.toPrecision(5)}`;this.hover.hidden=false;this.hover.style.left=Math.min(event.clientX-rect.left+12,rect.width-205)+'px';this.hover.style.top=Math.max(8,event.clientY-rect.top-35)+'px';}
  tick(){if(!this.running)return;requestAnimationFrame(()=>this.tick());this.controls.update();this.renderer.render(this.scene,this.camera);}
}
