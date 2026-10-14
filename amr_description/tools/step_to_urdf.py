"""Build link meshes and mass properties from a step_dump.py cache.

Usage:
  python step_dump.py Product1_V0.2.stp /tmp/dump          # needs cadquery-ocp
  python step_to_urdf.py /tmp/dump links.yaml ../meshes ../urdf/amr_inertials.xacro

Writes one full-resolution COLLADA mesh per link (link frame, metres, CAD colours
as materials, no decimation) and a xacro file with an <inertial> macro per link.
Joint origins are taken from links.yaml 'frames'.
"""

import datetime
import io
import json
import re
import sys

import collada
import numpy as np
import trimesh
import yaml


def write_dae(path, by_color):
    """Write a COLLADA file with one geometry and Phong material per colour."""
    dae = collada.Collada()
    dae.assetInfo.unitmeter = 1.0
    dae.assetInfo.unitname = 'meter'  # some loaders (Omniverse) require <unit>
    dae.assetInfo.upaxis = collada.asset.UP_AXIS.Z_UP
    # Fixed dates so regenerating unchanged geometry gives identical files (clean git diffs).
    dae.assetInfo.created = dae.assetInfo.modified = datetime.datetime(2026, 1, 1)
    nodes, ntri = [], 0
    for i, (col, chunks) in enumerate(sorted(by_color.items())):
        vs, fs, off = [], [], 0
        for v, f in chunks:
            used = np.unique(f)
            remap = np.full(len(v), -1, np.int64)
            remap[used] = np.arange(len(used))
            vs.append(v[used])
            fs.append(remap[f] + off)
            off += len(used)
        mesh = trimesh.Trimesh(np.concatenate(vs), np.concatenate(fs), process=False)
        verts = mesh.vertices.astype(np.float32).ravel()
        norms = mesh.vertex_normals.astype(np.float32).ravel()
        effect = collada.material.Effect(f'effect{i}', [], 'phong',
                                         diffuse=(*col, 1.0), ambient=(*col, 1.0),
                                         specular=(0.2, 0.2, 0.2, 1.0), shininess=20.0)
        mat = collada.material.Material(f'material{i}', f'material{i}', effect)
        dae.effects.append(effect)
        dae.materials.append(mat)
        geom = collada.geometry.Geometry(dae, f'geometry{i}', f'geometry{i}', [
            collada.source.FloatSource(f'verts{i}', verts, ('X', 'Y', 'Z')),
            collada.source.FloatSource(f'norms{i}', norms, ('X', 'Y', 'Z'))])
        inputs = collada.source.InputList()
        inputs.addInput(0, 'VERTEX', f'#verts{i}')
        inputs.addInput(0, 'NORMAL', f'#norms{i}')
        tris = geom.createTriangleSet(mesh.faces.astype(np.int32).ravel(), inputs, f'material{i}')
        geom.primitives.append(tris)
        dae.geometries.append(geom)
        matnode = collada.scene.MaterialNode(f'material{i}', mat, inputs=[])
        nodes.append(collada.scene.GeometryNode(geom, [matnode]))
        ntri += len(mesh.faces)
    scene = collada.scene.Scene('scene', [collada.scene.Node('node', children=nodes)])
    dae.scenes.append(scene)
    dae.scene = scene
    buf = io.BytesIO()
    dae.write(buf)
    # pycollada prints the full float repr; 5 significant digits (10 um) is plenty
    # and shrinks the file.
    text = re.sub(rb'(<float_array[^>]*>)([^<]*)', _compact_floats, buf.getvalue())
    open(path, 'wb').write(text)
    return ntri


def _compact_floats(match):
    values = np.array(match.group(2).split(), dtype=np.float64)
    return match.group(1) + ' '.join(f'{x:.5g}' for x in values).encode()


def main():
    dump_dir, cfg_path, mesh_dir, xacro_out = sys.argv[1:5]
    cfg = yaml.safe_load(open(cfg_path))
    parts = json.load(open(f'{dump_dir}/parts.json'))
    meshes = np.load(f'{dump_dir}/meshes.npz')

    def match_link(path):
        for link, keys in cfg['links'].items():
            if any(k in path for k in keys):
                return link
        return 'base_link'

    def match_rule(path):
        for i, rule in enumerate(cfg['mass_rules']):
            if rule['match'] in path:
                return i
        raise ValueError(path)

    # Assign link and mass rule to every leaf part.
    for p in parts:
        p['spath'] = ' / '.join(p['path'])
        p['link'] = match_link(p['spath'])
        p['rule'] = match_rule(p['spath'])

    # Group volumes per rule for 'mass' rules.
    rule_vol = {}
    for p in parts:
        if p['volume'] > 1.0:
            rule_vol[p['rule']] = rule_vol.get(p['rule'], 0.0) + p['volume']
    for p in parts:
        rule = cfg['mass_rules'][p['rule']]
        if p['volume'] <= 1.0:
            p['mass'] = 0.0
        elif 'mass' in rule:
            p['mass'] = rule['mass'] * p['volume'] / rule_vol[p['rule']]
        else:
            p['mass'] = rule['density'] * 1e-6 * p['volume']  # g/cm3 -> kg/mm3

    total = sum(p['mass'] for p in parts)
    scale = 1.0
    if cfg.get('measured_total_mass'):
        scale = cfg['measured_total_mass'] / total
    for p in parts:
        p['mass'] *= scale

    # Combine mass properties per link, expressed in the link frame (metres).
    links = {}
    for p in parts:
        if p['mass'] <= 0.0:
            continue
        origin = np.array(cfg['frames'][p['link']])
        c = (np.array(p['centroid']) - origin) * 1e-3
        rho = p['mass'] / p['volume']  # kg/mm3
        inertia = np.array(p['inertia_about_com_unit_density']) * rho * 1e-6  # kg m2
        e = links.setdefault(p['link'], {'m': 0.0, 'mc': np.zeros(3), 'parts': []})
        e['m'] += p['mass']
        e['mc'] += p['mass'] * c
        e['parts'].append((p['mass'], c, inertia))
    for e in links.values():
        com = e['mc'] / e['m']
        tensor = np.zeros((3, 3))
        for m, c, inertia in e['parts']:
            d = c - com
            tensor += inertia + m * (d.dot(d) * np.eye(3) - np.outer(d, d))
        e['com'], e['I'] = com, tensor

    # Export full-resolution visual meshes (COLLADA, one material per CAD colour).
    # Vertices are only shared inside a CAD face, so normals stay sharp at edges.
    for link in links:
        origin = np.array(cfg['frames'][link])
        by_color = {}
        for p in parts:
            if p['link'] != link:
                continue
            v, f, c = (meshes[f"{k}{p['id']}"] for k in 'vfc')
            if len(f) == 0:
                continue
            v = (v.astype(np.float64) - origin) * 1e-3
            for ci, col in enumerate(p['palette']):
                sel = f[c == ci]
                if len(sel):
                    by_color.setdefault(tuple(col), []).append((v, sel))
        ntri = write_dae(f'{mesh_dir}/{link}.dae', by_color)
        print(f'{link:18s} faces={ntri:8d} colours={len(by_color):3d} '
              f'mass={links[link]["m"]:.3f} kg com={np.round(links[link]["com"], 4)}')

    # Write inertials xacro.
    out = ['<?xml version="1.0"?>',
           '<!-- GENERATED by tools/step_to_urdf.py from links.yaml. Do not edit by hand. -->',
           '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">']
    for link, e in sorted(links.items()):
        c, inertia = e['com'], e['I']
        out += [f'  <xacro:macro name="inertial_{link}">',
                '    <inertial>',
                f'      <origin xyz="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" rpy="0 0 0"/>',
                f'      <mass value="{e["m"]:.4f}"/>',
                f'      <inertia ixx="{inertia[0, 0]:.6e}" ixy="{inertia[0, 1]:.6e}" '
                f'ixz="{inertia[0, 2]:.6e}" iyy="{inertia[1, 1]:.6e}" '
                f'iyz="{inertia[1, 2]:.6e}" izz="{inertia[2, 2]:.6e}"/>',
                '    </inertial>',
                '  </xacro:macro>']
    out.append('</robot>')
    open(xacro_out, 'w').write('\n'.join(out) + '\n')

    # Mass report.
    print(f'\ntotal mass {sum(e["m"] for e in links.values()):.3f} kg (scale {scale:.3f})')
    by_rule = {}
    for p in parts:
        by_rule[p['rule']] = by_rule.get(p['rule'], 0.0) + p['mass']
    for i, m in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        r = cfg['mass_rules'][i]
        print(f'  {m:7.3f} kg  {r["match"] or "<default>":45s} {r.get("note", "")}')


if __name__ == '__main__':
    main()
