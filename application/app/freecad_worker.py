"""Stdlib-only script executed by FreeCAD's Python, not the web virtualenv."""
import json
import os
from pathlib import Path
import sys
import struct
import traceback


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def build(intent, output):
    import FreeCAD as App
    import Part
    doc = App.newDocument("DrawingModel")
    shapes = {}
    objects = {}
    dimension_checks=[]
    history=[]
    try:
        for op in intent["operations"]:
            kind = op["type"]
            origin = App.Vector(*op.get("origin",[0,0,0]))
            direction=App.Vector(*op.get('direction',[0,0,1]));direction.normalize()
            write_json(output/'active-operation.json',{'operation':op,'state':'running'})
            retry=None
            if kind == "cylinder":
                shape = Part.makeCylinder(op["radius"],op["height"],origin,direction)
            elif kind == "box":
                shape = Part.makeBox(op["length"],op["width"],op["height"],origin,direction)
            elif kind in ("revolve","extrude"):
                if kind == "revolve":
                    pts = [App.Vector(r,0,z) for r,z in op["profile"]]
                else:
                    pts = [App.Vector(x,y,0) for x,y in op["profile"]]
                geometries=[]
                if op.get('profile_segments'):
                    for segment in op['profile_segments']:
                        values=[App.Vector(x,0,y) if kind=='revolve' else App.Vector(x,y,0) for x,y in segment['points']]
                        curve=Part.LineSegment(*values) if segment['kind']=='line' else Part.Arc(*values)
                        geometries.append(curve)
                    wire=Part.Wire([curve.toShape() for curve in geometries])
                else:wire = Part.makePolygon(pts+[pts[0]])
                face = Part.Face(wire)
                if not face.isValid():
                    raise ValueError("Невалидный профиль: проверьте самопересечения")
                face.check(True)
                shape = face.revolve(App.Vector(0,0,0),App.Vector(0,0,1),op.get("angle",360)) if kind=="revolve" else face.extrude(App.Vector(0,0,op["height"]))
                shape.Placement=App.Placement(origin,App.Rotation(App.Vector(0,0,1),direction))
            elif kind == "cut":
                try:
                    shape = shapes[op["base"]].cut(shapes[op["tool"]])
                    shape.check(True)
                except Exception:
                    # Same operands and dimensions, with an OCC fuzzy tolerance
                    # far below the dimension-check tolerance. No substitute body.
                    shape=shapes[op['base']].removeSplitter().cut(shapes[op['tool']].removeSplitter(),1e-7)
                    retry='same_boolean_operands_refined_and_fuzzy_1e-7_mm'
            elif kind == "fuse":
                shape = shapes[op["base"]].fuse(shapes[op["tool"]])
            elif kind == "hole":
                shape = shapes[op["base"]].cut(Part.makeCylinder(op["radius"],op["height"],origin,direction))
            elif kind == 'circular_pattern':
                tools=[]
                for index in range(op['count']):
                    copy=shapes[op['tool']].copy()
                    copy.rotate(origin,direction,op.get('angle',360)*index/op['count'])
                    tools.append(copy)
                shape=shapes[op['base']].cut(Part.makeCompound(tools))
            elif kind in ('chamfer','fillet'):
                base=shapes[op['base']]
                if op.get('edge_selector'):
                    selector=op['edge_selector'];point=App.Vector(*selector['point'])
                    ranked=sorted((edge.distToShape(Part.Vertex(point))[0],index) for index,edge in enumerate(base.Edges,1))
                    if not ranked or ranked[0][0]>selector['tolerance_mm']:raise ValueError('No edge near the declared image feature')
                    # Ambiguous equal-distance edges are not all rounded silently.
                    if len(ranked)>1 and abs(ranked[1][0]-ranked[0][0])<1e-5:raise ValueError('Ambiguous geometric edge selector')
                    op['edges']=[ranked[0][1]]
                if any(i>len(base.Edges) for i in op['edges']):raise ValueError('Номер ребра вне диапазона')
                edges=[base.Edges[i-1] for i in op['edges']]
                shape=base.makeChamfer(op['distance'],edges) if kind=='chamfer' else base.makeFillet(op['radius'],edges)
            else:
                raise ValueError("Unsupported operation")
            if shape.isNull() or not shape.isValid() or not shape.Solids or shape.Volume <= 1e-8:
                raise ValueError(f"Операция {op['id']} не создала валидное твёрдое тело")
            if kind in ('hole','cut','circular_pattern') and shape.Volume>=shapes[op['base']].Volume-1e-8:
                raise ValueError('Заявленный вырез не удалил материал из базового тела')
            shape.check(True)
            shapes[op["id"]] = shape
            # Native FreeCAD features retain editable primitive parameters and links.
            if kind in ('cylinder','box'):
                node=doc.addObject('Part::Cylinder' if kind=='cylinder' else 'Part::Box','Op_'+op['id'])
                if kind=='cylinder':node.Radius=op['radius'];node.Height=op['height']
                else:node.Length=op['length'];node.Width=op['width'];node.Height=op['height']
                node.Placement.Base=origin
                node.Placement.Rotation=App.Rotation(App.Vector(0,0,1),direction)
            elif kind in ('revolve','extrude'):
                if kind=='extrude' and (intent.get('evidence',{}).get('independent_semantic_graph') or intent.get('evidence',{}).get('analytic_profile')):
                    import Sketcher
                    source=doc.addObject('Sketcher::SketchObject','InputProfile_'+op['id'])
                    if geometries:
                        for curve in geometries:source.addGeometry(curve,False)
                    else:
                        for a,b in zip(pts,pts[1:]+pts[:1]):source.addGeometry(Part.LineSegment(a,b),False)
                    doc.recompute()
                else:
                    source=doc.addObject('Part::Feature','InputProfile_'+op['id']);source.Shape=face
                node=doc.addObject('Part::Revolution' if kind=='revolve' else 'Part::Extrusion','Op_'+op['id'])
                if kind=='revolve':
                    node.Source=source;node.Axis=App.Vector(0,0,1);node.Base=App.Vector(0,0,0);node.Angle=op.get('angle',360);node.Solid=True
                else:
                    node.Base=source;node.DirMode='Custom';node.Dir=App.Vector(0,0,1);node.LengthFwd=op['height'];node.Solid=True
                node.Placement.Base=origin
                node.Placement.Rotation=App.Rotation(App.Vector(0,0,1),direction)
            elif kind in ('cut','hole','fuse','circular_pattern'):
                if kind=='hole':
                    tool=doc.addObject('Part::Cylinder','HoleTool_'+op['id']);tool.Radius=op['radius'];tool.Height=op['height'];tool.Placement.Base=origin
                    tool.Placement.Rotation=App.Rotation(App.Vector(0,0,1),direction)
                elif kind=='circular_pattern':
                    instances=[]
                    for index in range(op['count']):
                        instance=doc.addObject('PartDesign::Feature','PatternInstance_'+op['id']+'_'+str(index))
                        instance.Shape=tools[index];instances.append(instance)
                    tool=doc.addObject('Part::Compound','PatternTools_'+op['id']);tool.Links=instances
                else:tool=objects[op['tool']]
                node=doc.addObject('Part::Fuse' if kind=='fuse' else 'Part::Cut','Op_'+op['id'])
                node.Base=objects[op['base']];node.Tool=tool
            else:
                node=doc.addObject('Part::Chamfer' if kind=='chamfer' else 'Part::Fillet','Op_'+op['id'])
                node.Base=objects[op['base']]
                value=op['distance'] if kind=='chamfer' else op['radius']
                # Native primitives and Part.make* can enumerate identical edges
                # differently. Match geometry before setting native selectors.
                selected=[]
                for original_index in op['edges']:
                    edge=shapes[op['base']].Edges[original_index-1]
                    bb=edge.BoundBox
                    signature=[bb.XMin,bb.YMin,bb.ZMin,bb.XMax,bb.YMax,bb.ZMax,edge.Length]
                    matching=[]
                    for j,native in enumerate(objects[op['base']].Shape.Edges,1):
                        box=native.BoundBox
                        other=[box.XMin,box.YMin,box.ZMin,box.XMax,box.YMax,box.ZMax,native.Length]
                        if type(edge.Curve).__name__==type(native.Curve).__name__ and max(abs(a-b) for a,b in zip(signature,other))<1e-5:matching.append(j)
                    if len(matching)!=1:raise ValueError('Не удалось однозначно сопоставить ребро нативной истории FreeCAD')
                    selected.append(matching[0])
                node.Edges=[(i,value,value) for i in selected]
            node.addProperty('App::PropertyString','OperationJSON','Evidence');node.OperationJSON=json.dumps(op,ensure_ascii=False)
            objects[op['id']]=node
            doc.recompute()
            if node.Shape.isNull() or not node.Shape.isValid() or abs(node.Shape.Volume-shape.Volume)>max(1e-6,shape.Volume*1e-7):
                raise ValueError(f"Нативная параметрическая операция {op['id']} не совпала с проверенным BRep")
            history.append({'id':op['id'],'type':kind,'state':'passed','volume_mm3':shape.Volume,'solid_count':len(shape.Solids),'retry':retry})
            write_json(output/'operation-history.json',history)
            write_json(output/'active-operation.json',{'operation':op,'state':'passed'})
        shape = shapes[intent["result"]].removeSplitter()
        shape.check(True)
        obj = doc.addObject("Part::Compound","Result")
        obj.Label = "Result ("+intent["result"]+")"
        obj.Links=[objects[intent['result']]]
        obj.addProperty("App::PropertyString","CADIntentJSON","Source")
        obj.CADIntentJSON = json.dumps(intent,ensure_ascii=False)
        doc.recompute()
        for item in doc.Objects:
            if hasattr(item,'Visibility'):item.Visibility=(item.Name=='Result')
        doc.saveAs(str(output/"model.FCStd"))
        # Persist both 3D edge curves and their surface parameter curves. The
        # runtime's default without pcurves can round-trip intersecting cylinders
        # into InvalidCurveOnSurface despite a valid source BRep. This changes
        # representation only, never geometry, tolerances or strict BOP checks.
        Part.setStaticValue('write.surfacecurve.mode',1)
        shape.exportStep(str(output/"model.step"))
        imported = Part.read(str(output/"model.step"))
        # isValid() alone can miss invalid curve/surface parameterizations
        # introduced by the STEP round trip at intersecting cylinders.
        imported.check(True)
        if not imported.isValid() or abs(imported.Volume-shape.Volume) > max(1e-6,shape.Volume*1e-7):
            raise ValueError("Проверка повторного чтения STEP не прошла")
        # Validate real BRep sections, not just the input parameters or bounding box.
        vertex_z=sorted(set(round(v.Point.z,7) for v in shape.Vertexes))
        for expected in intent.get('validation_expectations',[]):
            check=dict(expected)
            if expected['type']=='analytic_profile_radius':
                radii=[edge.Curve.Radius for edge in shape.Edges if hasattr(edge.Curve,'Radius')]
                radii += [face.Surface.MinorRadius for face in shape.Faces if hasattr(face.Surface,'MinorRadius')]
                observed=min(radii,key=lambda radius:abs(radius-expected['nominal_mm'])) if radii else None
                check.update(status='passed' if observed is not None and abs(observed-expected['nominal_mm'])<.03 else 'failed',
                             measured_mm=observed,error_mm=observed-expected['nominal_mm'] if observed is not None else None)
            elif expected['type']=='semantic_cylinder':
                direction=App.Vector(*expected['direction']);direction.normalize();origin=App.Vector(*expected['origin'])
                candidates=[]
                for face in shape.Faces:
                    surface=face.Surface
                    if not all(hasattr(surface,k) for k in ('Radius','Axis','Center')):continue
                    if abs(abs(surface.Axis.dot(direction))-1)>.001:continue
                    distance=(surface.Center-origin).cross(direction).Length
                    if distance>.03:continue
                    axial=[(v.Point-origin).dot(direction) for v in face.Vertexes]
                    span=max(axial)-min(axial) if axial else 0
                    candidates.append({'radius_error_mm':surface.Radius-expected['radius_mm'],
                                       'axial_extent_mm':span,'length_error_mm':span-expected['length_mm']})
                chosen=min(candidates,key=lambda c:abs(c['radius_error_mm'])+abs(c['length_error_mm'])) if candidates else None
                check.update(status='passed' if chosen and abs(chosen['radius_error_mm'])<.03 and abs(chosen['length_error_mm'])<.03 else 'failed',
                             measured=chosen)
            elif expected['type']=='semantic_void':
                origin=App.Vector(*expected['origin']);axis=App.Vector(*expected['direction']);axis.normalize()
                radial=axis.cross(App.Vector(0,1,0))
                if radial.Length<.1:radial=axis.cross(App.Vector(1,0,0))
                radial.normalize();checks=[]
                for fraction in (.15,.5,.85):
                    point=origin+axis*(expected['depth_mm']*fraction)
                    checks.append({'fraction':fraction,'void':not shape.isInside(point,1e-6,True),
                                   'surrounding_material':shape.isInside(point+radial*expected['surrounding_radius_mm'],1e-6,True)})
                check.update(status='passed' if all(c['void'] and c['surrounding_material'] for c in checks) else 'failed',samples=checks)
            elif expected['type']=='axial_distance':
                if not expected.get('grounded',True):
                    check.update(status='unverified_disconnected_chain');dimension_checks.append(check);continue
                requested=expected['positions_mm']
                actual=[min(vertex_z,key=lambda z:abs(z-x)) for x in requested]
                if max(abs(a-b) for a,b in zip(actual,requested))>.35:
                    check.update(status='unverified_no_topological_plane',actual_positions_mm=actual)
                else:
                    measured=abs(actual[1]-actual[0]);error=measured-expected['nominal_mm']
                    check.update(status='passed' if abs(error)<.03 else 'failed',measured_mm=measured,error_mm=error)
            elif expected['type']=='hole_pattern':
                observed=[]
                for wire in shape.slice(App.Vector(0,0,1),expected['z_mm']):
                    for edge in wire.Edges:
                        curve=edge.Curve
                        if hasattr(curve,'Radius') and hasattr(curve,'Center'):
                            observed.append((curve.Center.x,curve.Center.y,curve.Radius))
                holes=[]
                for x,y in expected['centers_xy_mm']:
                    errors=[((a-x)**2+(b-y)**2)**.5+abs(r-expected['radius_mm']) for a,b,r in observed]
                    holes.append(min(errors) if errors else None)
                check.update(status='passed' if all(e is not None and e<.03 for e in holes) else 'failed',
                             center_and_radius_errors_mm=holes)
            elif expected['type']=='bounding_extent':
                axis=expected['axis'];box=shape.BoundBox
                measured=[box.XLength,box.YLength,box.ZLength][axis]
                error=measured-expected['nominal_mm']
                check.update(status='passed' if abs(error)<.03 else 'failed',measured_mm=measured,error_mm=error)
            elif expected['type']=='coaxial_bore_depth':
                matches=[]
                for face in shape.Faces:
                    surface=face.Surface
                    if not hasattr(surface,'Radius') or not hasattr(surface,'Axis') or not hasattr(surface,'Center'):continue
                    if abs(surface.Radius-expected['radius_mm'])>.03 or abs(abs(surface.Axis.z)-1)>.001:continue
                    if abs(surface.Center.x)+abs(surface.Center.y)>.03:continue
                    box=face.BoundBox
                    matches.append((box.ZMin,box.ZMax))
                nominal=(expected['start_z_mm'],expected['start_z_mm']+expected['depth_mm'])
                errors=[max(abs(a-nominal[0]),abs(b-nominal[1])) for a,b in matches]
                error=min(errors) if errors else None
                check.update(status='passed' if error is not None and error<.03 else 'failed',
                             axial_interval_error_mm=error,observed_intervals_mm=matches)
            elif expected['type']=='cylindrical_surface':
                checks=[]
                for lo,hi in expected['axial_intervals_mm']:
                    z=(lo+hi)/2
                    wires=shape.slice(App.Vector(0,0,1),z)
                    radii=[]
                    for wire in wires:
                        for edge in wire.Edges:
                            if hasattr(edge.Curve,'Radius'):radii.append(edge.Curve.Radius)
                    if radii:
                        measured=min((2*r for r in radii),key=lambda d:abs(d-expected['nominal_mm']))
                        checks.append({'z_mm':z,'measured_mm':measured,'error_mm':measured-expected['nominal_mm']})
                check.update(status='passed' if checks and max(abs(c['error_mm']) for c in checks)<.03 else 'failed' if checks else 'unverified_no_surface',sections=checks)
            dimension_checks.append(check)
        dimensions_ok=not any(x.get('status')=='failed' and x.get('required',True) for x in dimension_checks)
        surfaces=[]
        for face in shape.Faces:
            surface=face.Surface
            if not all(hasattr(surface,key) for key in ('Radius','Axis','Center')):continue
            axis=surface.Axis;center=surface.Center;axis.normalize();role='unknown'
            try:
                u0,u1,v0,v1=face.ParameterRange;point=face.valueAt((u0+u1)/2,(v0+v1)/2)
                foot=center+axis*((point-center).dot(axis));radial=point-foot;radial.normalize()
                epsilon=min(.02,max(.0001,surface.Radius*.001))
                inside=shape.isInside(foot+radial*(surface.Radius-epsilon),1e-7,True)
                outside=shape.isInside(foot+radial*(surface.Radius+epsilon),1e-7,True)
                role='outer' if inside and not outside else 'inner' if outside and not inside else 'unknown'
            except Exception:pass
            positions=[(vertex.Point-center).dot(axis) for vertex in face.Vertexes]
            surfaces.append({'radius_mm':surface.Radius,'axis':[axis.x,axis.y,axis.z],
                'center_mm':[center.x,center.y,center.z],'role':role,
                'axial_span_mm':max(positions)-min(positions) if positions else 0})
        axis_planes={str(axis):sorted(set(round(getattr(vertex.Point,name),6) for vertex in shape.Vertexes))
                     for axis,name in enumerate(('x','y','z'))}
        validation={'geometry':{'valid':shape.isValid(),'volume_positive':shape.Volume>0,'solid_count':len(shape.Solids),
                    'step_reopened':True,'step_volume_error_mm3':imported.Volume-shape.Volume,
                    'bounding_extents_mm':[shape.BoundBox.XLength,shape.BoundBox.YLength,shape.BoundBox.ZLength]},
                    'cylindrical_surface_inventory':surfaces,'topological_planes_by_axis_mm':axis_planes,
                    'dimensions':dimension_checks,'dimension_checks_passed':dimensions_ok,'required_dimension_checks_passed':dimensions_ok,
                    'nonblocking_dimension_conflicts':[x for x in dimension_checks if x.get('status')=='failed' and not x.get('required',True)],
                    'reconstruction_status':intent.get('reconstruction_status','manual')}
        write_json(output/'validation-brep.json',validation)
        if not dimensions_ok:raise ValueError('Построенное тело не прошло проверку связанных размеров')
        vertices,triangles = shape.tessellate(.05)
        if len(vertices)>500000 or len(triangles)>1000000:
            raise ValueError("Слишком большая сетка предпросмотра")
        with (output/'model.stl').open('wb') as stream:
            stream.write(b'Drawing-to-3D FreeCAD tessellation; coordinates mm'.ljust(80,b' '))
            stream.write(struct.pack('<I',len(triangles)))
            for triangle in triangles:
                points=[vertices[i] for i in triangle]
                normal=(points[1]-points[0]).cross(points[2]-points[0])
                if normal.Length>1e-12:normal.normalize()
                values=[normal.x,normal.y,normal.z]+[v for p in points for v in (p.x,p.y,p.z)]
                stream.write(struct.pack('<12fH',*values,0))
        edges = []
        for edge in shape.Edges:
            points = edge.discretize(Deflection=.08)
            for a,b in zip(points,points[1:]):
                edges.extend([[a.x,a.y,a.z],[b.x,b.y,b.z]])
        preview = {"units":"mm","vertices":[[v.x,v.y,v.z] for v in vertices],"triangles":[list(t) for t in triangles],
                   "edges":edges,"cad_engine":"FreeCAD","freecad_version":".".join(App.Version()[:3]),
                   "volume_mm3":shape.Volume,"solid_count":len(shape.Solids),"valid":shape.isValid(),
                   "bounds":{"min":[shape.BoundBox.XMin,shape.BoundBox.YMin,shape.BoundBox.ZMin],
                             "max":[shape.BoundBox.XMax,shape.BoundBox.YMax,shape.BoundBox.ZMax]},
                   "provenance":intent["provenance"],"description":intent["description"],
                   "reconstruction_status":intent.get('reconstruction_status','manual'),
                   "unresolved_features":intent.get('unresolved_features',[]),
                   "operation_history":[x['id'] for x in intent['operations']],
                   "section_xz_wires":[[[p.x,p.z] for p in wire.discretize(Deflection=.025)] for wire in shape.slice(App.Vector(0,1,0),0)]}
        write_json(output/"preview.json",preview)
        return {k:v for k,v in preview.items() if k not in ("vertices","triangles","edges")}
    finally:
        App.closeDocument(doc.Name)


def main():
    output = Path(os.environ["DRAWING_CAD_OUTPUT"])
    output.mkdir(parents=True,exist_ok=True)
    try:
        intent = json.loads(Path(os.environ["DRAWING_CAD_INTENT"]).read_text(encoding="utf-8"))
        result = build(intent,output)
        write_json(output/"result.json",{"ok":True,**result})
    except Exception as exc:
        context=json.loads((output/'active-operation.json').read_text(encoding='utf-8')) if (output/'active-operation.json').is_file() else None
        write_json(output/"result.json",{"ok":False,"error":str(exc),"failed_operation":context,"traceback":traceback.format_exc()})
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    # FreeCADCmd can otherwise enter its interactive console after executing a file.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(code)
