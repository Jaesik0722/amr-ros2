"""Load a STEP assembly once and dump every leaf instance (world frame, mm).

Usage (needs cadquery-ocp):
  python step_dump.py Product1_V0.2.stp OUT_DIR [--config links.yaml]

Own designed parts are tessellated finely; purchased parts matched by the
'tessellation' groups in links.yaml (NUC, cameras, hub motors, ...) use a
coarser tolerance, and small internal parts and solids (SMD parts, connectors,
pins) can be dropped from the visual mesh, so the meshes stay a manageable size. Mass properties
always use the full B-rep.

Outputs in OUT_DIR:
  parts.json  - per leaf: id, path (instance/product names), bbox, volume, centroid,
                inertia, and the palette of CAD colours used by its faces
  meshes.npz  - per leaf: v<id> (N,3 float32 mm), f<id> (M,3 int32),
                c<id> (M,) uint8 index into the part's palette (per-face CAD colour)
"""
import argparse
import json
import time

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.XCAFDoc import (XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool,
                         XCAFDoc_DocumentTool)
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('step')
ap.add_argument('out_dir')
ap.add_argument('--config', help="links.yaml with a 'tessellation' section")
args = ap.parse_args()
step_path, out_dir = args.step, args.out_dir
tess = {'default': [0.1, 0.2], 'groups': []}  # tolerances: [linear mm, angular rad]
if args.config:
    tess.update(yaml.safe_load(open(args.config)).get('tessellation', {}))


def tess_for(spath):
    for g in tess['groups']:
        if any(k in spath for k in g['match']):
            return g['tol'][0], g['tol'][1], g.get('min_size_mm', 0.0)
    return tess['default'][0], tess['default'][1], 0.0


DEFAULT_RGB = (0.7, 0.7, 0.7)

t0 = time.time()
doc = TDocStd_Document(TCollection_ExtendedString('doc'))
reader = STEPCAFControl_Reader()
reader.SetNameMode(True)
reader.SetColorMode(True)
assert reader.ReadFile(step_path) == 1, 'read failed'
reader.Transfer(doc)
print(f'loaded in {time.time() - t0:.0f}s', flush=True)

st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())


def name_of(label):
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ''


def rgb(col):
    r, g, b = col.Values(Quantity_TOC_sRGB)
    return (round(r, 4), round(g, 4), round(b, 4))


def label_color(label):
    col = Quantity_Color()
    for ctype in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        if XCAFDoc_ColorTool.GetColor_s(label, ctype, col):
            return rgb(col)
    return None


def face_color(face):
    col = Quantity_Color()
    for ctype in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        if ct.GetColor(face, ctype, col):
            return rgb(col)
    return None


parts = []
meshes = {}


def small_solid_faces(shape, min_size):
    """Faces of solids whose bounding box diagonal is below min_size (mm)."""
    skip = TopTools_IndexedMapOfShape()
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        box = Bnd_Box()
        BRepBndLib.Add_s(exp.Current(), box)
        if not box.IsVoid() and box.CornerMin().Distance(box.CornerMax()) < min_size:
            TopExp.MapShapes_s(exp.Current(), TopAbs_FACE, skip)
        exp.Next()
    return skip


def mesh_shape(shape, trsf, base_rgb, lin, ang, min_size):
    """Triangulate the un-located shape (so face colour lookup works), then place it."""
    verts, faces, cols, palette, off = [], [], [], {}, 0
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if min_size > 0 and box.CornerMin().Distance(box.CornerMax()) < min_size:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                np.zeros(0, np.uint8), [])
    BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    skip = small_solid_faces(shape, min_size) if min_size > 0 else TopTools_IndexedMapOfShape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if skip.Contains(face):
            exp.Next()
            continue
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            ftrsf = trsf.Multiplied(loc.Transformation())
            n = tri.NbNodes()
            for i in range(1, n + 1):
                p = tri.Node(i).Transformed(ftrsf)
                verts.append((p.X(), p.Y(), p.Z()))
            ci = palette.setdefault(face_color(face) or base_rgb, len(palette))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                faces.append((a - 1 + off, c - 1 + off, b - 1 + off) if rev
                             else (a - 1 + off, b - 1 + off, c - 1 + off))
                cols.append(ci)
            off += n
        exp.Next()
    return (np.asarray(verts, np.float32).reshape(-1, 3),
            np.asarray(faces, np.int32).reshape(-1, 3),
            np.asarray(cols, np.uint8), [list(c) for c in palette])


def walk(label, loc, path, inst_rgb, parent_rgb):
    # Colour precedence: instance > product > enclosing assembly > default.
    own = label_color(label)
    if st.IsAssembly_s(label):
        comps = TDF_LabelSequence()
        st.GetComponents_s(label, comps, False)
        for i in range(1, comps.Length() + 1):
            comp = comps.Value(i)
            ref = TDF_Label()
            st.GetReferredShape_s(comp, ref)
            cl = st.GetLocation_s(comp)
            walk(ref, loc.Multiplied(cl), path + [f'{name_of(comp)}|{name_of(ref)}'],
                 label_color(comp), inst_rgb or own or parent_rgb)
        return
    shape = st.GetShape_s(label)
    if shape.IsNull():
        return
    placed = shape.Moved(loc)
    pid = len(parts)
    box = Bnd_Box()
    BRepBndLib.Add_s(placed, box)
    if box.IsVoid():
        return
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(placed, props)
    vol = props.Mass()
    com = props.CentreOfMass()
    m = props.MatrixOfInertia()
    inertia = [[m.Value(r, c) for c in range(1, 4)] for r in range(1, 4)]
    base_rgb = inst_rgb or own or parent_rgb or DEFAULT_RGB
    v, f, c, palette = mesh_shape(shape, loc.Transformation(), base_rgb,
                                  *tess_for(' / '.join(path)))
    meshes[f'v{pid}'] = v
    meshes[f'f{pid}'] = f
    meshes[f'c{pid}'] = c
    parts.append({
        'id': pid, 'path': path, 'bbox': [xmin, ymin, zmin, xmax, ymax, zmax],
        'volume': vol, 'centroid': [com.X(), com.Y(), com.Z()],
        'inertia_about_com_unit_density': inertia, 'ntri': int(len(f)), 'palette': palette,
    })
    if pid % 50 == 0:
        print(f'{pid} parts, {time.time() - t0:.0f}s', flush=True)


free = TDF_LabelSequence()
st.GetFreeShapes(free)
for i in range(1, free.Length() + 1):
    lab = free.Value(i)
    walk(lab, TopLoc_Location(), [f'|{name_of(lab)}'], None, None)

json.dump(parts, open(f'{out_dir}/parts.json', 'w'), indent=1)
np.savez_compressed(f'{out_dir}/meshes.npz', **meshes)
print(f'done: {len(parts)} leaf parts, {sum(p["ntri"] for p in parts)} triangles '
      f'in {time.time() - t0:.0f}s')
