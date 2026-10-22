// This graph is reconstructed from the user's Keras file; no training takes place here.
export async function loadNetwork(tf, base = './', progress = () => {}, local = null) {
  if(!local) {
    const config=await fetch(new URL('model/network.json', new URL(base, globalThis.location?.href)));
    if(!config.ok) throw new Error('기본 모델 정보를 불러오지 못했습니다. 새로고침하거나 모델 파일을 직접 선택해 주세요.');
    const weights=await fetch(new URL('model/weights.bin', new URL(base, globalThis.location?.href)));
    if(!weights.ok) throw new Error('기본 모델 가중치를 불러오지 못했습니다. 새로고침하거나 모델 파일을 직접 선택해 주세요.');
    local={meta:await config.json(),buffer:await weights.arrayBuffer()};
  }
  const {meta,buffer}=local;
  validateModel(meta, buffer);
  const values = new Map(), objects = new Map();
  for (const [i, layer] of meta.layers.entries()) {
    const c = layer.config;
    let obj, output;
    if (layer.type === 'InputLayer') {
      output = tf.input({shape: meta.inputShape, name: layer.name});
    } else {
      const common = {name: layer.name, trainable: false};
      if (layer.type === 'Conv2D') obj = tf.layers.conv2d({...common, filters:c.filters,
        kernelSize:c.kernel_size, strides:c.strides, padding:c.padding,
        dataFormat:'channelsLast', dilationRate:c.dilation_rate, activation:c.activation, useBias:c.use_bias});
      else if (layer.type === 'MaxPooling2D') obj = tf.layers.maxPooling2d({...common,
        poolSize:c.pool_size, strides:c.strides, padding:c.padding, dataFormat:'channelsLast'});
      else if (layer.type === 'Flatten') obj = tf.layers.flatten({...common, dataFormat:'channelsLast'});
      else if (layer.type === 'Dense') obj = tf.layers.dense({...common, units:c.units,
        activation:c.activation, useBias:c.use_bias});
      else throw new Error('지원하지 않는 모델 연산: ' + layer.type);
      output = obj.apply(values.get(layer.inputs[0]));
      const tensors = meta.weights.filter(w => w.layer === layer.name).map(w =>
        tf.tensor(new Float32Array(buffer, w.offset, w.length), w.shape));
      if (tensors.length) { obj.setWeights(tensors); tensors.forEach(t => t.dispose()); }
      objects.set(layer.name, obj);
    }
    values.set(layer.name, output);
    progress((i+1)/meta.layers.length);
  }
  const input = values.get(meta.layers[0].name);
  const model = tf.model({inputs:input, outputs:meta.layers.map(l=>values.get(l.name))});
  const weights = new Map(meta.weights.map(w=>[w.name, new Float32Array(buffer,w.offset,w.length)]));
  return {tf, meta, model, weights, async infer(pixels) {
    const inputTensor = tf.tensor4d(pixels, [1,...meta.inputShape]);
    let tensors;
    try {
      tensors = model.predict(inputTensor);
      const outputs = new Map();
      // Copy outputs once. Rendering and replay need no additional inference.
      for (let i=0;i<tensors.length;i++) outputs.set(meta.layers[i].name,
        {shape:tensors[i].shape.slice(1), data:new Float32Array(await tensors[i].data())});
      return outputs;
    } finally { inputTensor.dispose(); tensors?.forEach(t=>t.dispose()); }
  }};
}

export function validateModel(meta, buffer) {
  const fail=()=>{throw new Error('이 프로젝트용 network.json과 weights.bin을 함께 선택해 주세요.');};
  if(meta?.format!=='hangul-h5-float32-v1'||!Array.isArray(meta.layers)||meta.layers.length!==55||
     JSON.stringify(meta.inputShape)!=='[64,64,3]'||!(buffer instanceof ArrayBuffer)||buffer.byteLength!==11006416)fail();
  const labels=[list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'),list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'),['없음',...list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')]];
  if(JSON.stringify(meta.labels)!==JSON.stringify(labels))fail();
  const names=new Set();let total=0;
  for(const l of meta.layers){
    if(!/^[A-Za-z0-9_]+$/.test(l.name)||names.has(l.name)||!Array.isArray(l.inputs)||l.inputs.some(n=>!names.has(n))||
       !['InputLayer','Conv2D','MaxPooling2D','Flatten','Dense'].includes(l.type)||l.name!==l.config?.name)fail();
    if(l.type==='Conv2D'&&(![16,32,64,128,256].includes(l.config.filters)||!['relu'].includes(l.config.activation)))fail();
    if(l.type==='Dense'&&![19,21,28].includes(l.config.units))fail();
    names.add(l.name);
  }
  if(!Array.isArray(meta.outputs)||meta.outputs.length!==3||meta.outputs.some(n=>!names.has(n))||!Array.isArray(meta.weights))fail();
  for(const w of meta.weights){
    if(!names.has(w.layer)||!new RegExp('^'+w.layer+'/(kernel|bias):0$').test(w.name)||!Array.isArray(w.shape)||
       w.shape.some(n=>!Number.isInteger(n)||n<1||n>4096)||w.length!==w.shape.reduce((a,b)=>a*b,1)||
       !Number.isInteger(w.offset)||w.offset!==total||w.offset+w.length*4>buffer.byteLength)fail();
    total+=w.length*4;
  }
  if(total!==buffer.byteLength)fail();
  function list(s){return Array.from(s);}
}

export function branchLayers(meta, branch=0) {
  const byName = new Map(meta.layers.map(l=>[l.name,l]));
  const chain = [];
  let layer = byName.get(meta.outputs[branch]);
  while (layer) { chain.unshift(layer); layer = byName.get(layer.inputs[0]); }
  return chain;
}

export function decode(meta, outputs) {
  const distributions=meta.outputs.map(n=>Array.from(outputs.get(n).data));
  const ranked=distributions.map(a=>a.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p));
  const [a,b,c]=ranked.map(r=>r[0].i);
  const candidates=[];
  for (const x of ranked[0].slice(0,5)) for (const y of ranked[1].slice(0,5)) for(const z of ranked[2].slice(0,5))
    candidates.push({char:String.fromCharCode(0xac00+(x.i*21+y.i)*28+z.i),score:x.p*y.p*z.p});
  candidates.sort((x,y)=>y.score-x.score);
  return {char:String.fromCharCode(0xac00+(a*21+b)*28+c),ranked,distributions,candidates:candidates.slice(0,5)};
}

export function convAt(network, outputs, name, y, x, channel, inputChannel=0) {
  const layer=network.meta.layers.find(l=>l.name===name), c=layer.config;
  const input=outputs.get(layer.inputs[0]);
  const [h,w,cin]=input.shape, cout=c.filters, [kh,kw]=c.kernel_size;
  const kernel=network.weights.get(name+'/kernel:0'), bias=network.weights.get(name+'/bias:0');
  const padY=c.padding==='same'?Math.floor(kh/2):0, padX=c.padding==='same'?Math.floor(kw/2):0;
  let sum=bias[channel];const patch=[],filter=[];
  for(let ky=0;ky<kh;ky++) for(let kx=0;kx<kw;kx++) {
    const iy=y*c.strides[0]+ky-padY, ix=x*c.strides[1]+kx-padX;
    for(let ci=0;ci<cin;ci++) {
      const value=iy>=0&&iy<h&&ix>=0&&ix<w?input.data[(iy*w+ix)*cin+ci]:0;
      const weight=kernel[((ky*kw+kx)*cin+ci)*cout+channel];sum+=value*weight;
      if(ci===inputChannel){patch.push(value);filter.push(weight);}
    }
  }
  return {patch,filter,kh,kw,cin,sum,bias:bias[channel],actual:outputs.get(name).data[(y*outputs.get(name).shape[1]+x)*cout+channel]};
}
