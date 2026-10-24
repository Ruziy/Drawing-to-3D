import * as THREE from 'three';
import {OrbitControls} from '/static/vendor/OrbitControls.js';

export function createViewer(container) {
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.setClearColor(0xf6f8fa,1);
  container.append(renderer.domElement);
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(42,1,.01,100000);
  camera.up.set(0,0,1);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;
  scene.add(new THREE.HemisphereLight(0xffffff,0x87909c,2.4));
  const light=new THREE.DirectionalLight(0xffffff,3.2);
  light.position.set(100,-80,160);scene.add(light);
  let group=null, size=10;
  const reset=()=>{
    const distance=size/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)))*1.5/Math.min(1,camera.aspect);
    camera.position.set(distance*.75,-distance,distance*.7);
    camera.near=Math.max(size/1000,.001);camera.far=Math.max(size*100,100);
    controls.target.set(0,0,0);camera.updateProjectionMatrix();controls.update();
  };
  const resize=()=>{
    const w=container.clientWidth,h=container.clientHeight;
    renderer.setSize(w,h);camera.aspect=w/Math.max(h,1);camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);resize();
  renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});
  const clear=()=>{
    if(group){scene.remove(group);group.traverse(o=>{o.geometry?.dispose();o.material?.dispose()});group=null}
  };
  return {reset,clear,load(data){
    clear();group=new THREE.Group();
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.vertices.flat(),3));
    geometry.setIndex(data.triangles.flat());geometry.computeVertexNormals();
    const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0x8eaabd,metalness:.35,roughness:.5,side:THREE.DoubleSide}));
    group.add(mesh);
    if(data.edges?.length){
      const eg=new THREE.BufferGeometry();eg.setAttribute('position',new THREE.Float32BufferAttribute(data.edges.flat(),3));
      group.add(new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:0x263e50,transparent:true,opacity:.65})));
    }
    const box=new THREE.Box3().setFromObject(group),center=box.getCenter(new THREE.Vector3()),extent=box.getSize(new THREE.Vector3());
    group.position.sub(center);size=Math.max(extent.x,extent.y,extent.z,1);scene.add(group);reset();
  }};
}
