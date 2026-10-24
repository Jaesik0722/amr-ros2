"""Check the bay clearance against a brute-force reference and on hand-computed cases.

Run with the isaaclab python: it needs torch but not Isaac Sim.

The previous version of this file carried its own copy of `_bay_gap` and compared it
against the lateral-offset form it replaced. Both of those were wrong in the same way
-- they measured deck corners only -- so the test agreed with the code and said nothing.
It imports the real function now, and scores it against a reference that samples the
deck's whole boundary rather than its corners.
"""
import ast
import math
import os
import sys
import types

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'amr_lab')


def _lift(path, names):
    """Pull named top-level definitions out of a module without importing it.

    `towing_env` imports Isaac Sim at module scope, which is not available outside the
    simulator's interpreter, and `asset` is only needed here for two numbers. Reading
    the source keeps the test scored against the code that actually runs, which is the
    thing the previous version of this file got wrong by holding its own copy.
    """
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            out.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                out.append(node)
            elif any(isinstance(t, ast.Tuple) for t in targets) and \
                    any(isinstance(e, ast.Name) and e.id in names
                        for t in targets if isinstance(t, ast.Tuple) for e in t.elts):
                out.append(node)
    return ast.Module(body=out, type_ignores=[])


_asset = types.ModuleType('asset')
exec(compile(_lift('asset.py', {'DECK_HALF_L', 'DECK_HALF_W'}), 'asset', 'exec'),
     _asset.__dict__)
_ns = {'torch': torch, 'asset': _asset}
exec(compile(_lift('towing_env.py', {'_deck_corners', '_bay_gap'}),
             'towing_env', 'exec'), _ns)
_bay_gap, _deck_corners = _ns['_bay_gap'], _ns['_deck_corners']


class Cfg:
    bay_width = 1.00
    bay_depth = 1.6
    bay_wall_thickness = 0.1


DECK_HALF_L, DECK_HALF_W = _asset.DECK_HALF_L, _asset.DECK_HALF_W
T, W, D = Cfg.bay_wall_thickness, Cfg.bay_width, Cfg.bay_depth
BOXES = ((-0.5 * D, 0.5 * (W + T), 0.5 * D, 0.5 * T),
         (-0.5 * D, -0.5 * (W + T), 0.5 * D, 0.5 * T),
         (-D - 0.5 * T, 0.0, 0.5 * T, 0.5 * (W + 2 * T)))


def corner_only_gap(px, py):
    """The form this test used to bless: deck corners against the wall boxes."""
    best = None
    for cx, cy, hx, hy in BOXES:
        dx = (px - cx).abs() - hx
        dy = (py - cy).abs() - hy
        out = torch.sqrt(torch.clamp(dx, min=0.0) ** 2 + torch.clamp(dy, min=0.0) ** 2)
        ins = torch.clamp(torch.maximum(dx, dy), max=0.0)
        sdf = (out + ins).min(dim=1).values
        best = sdf if best is None else torch.minimum(best, sdf)
    return best


def reference(pos, yaw, steps=400):
    """Distance from the deck's sampled boundary to the wall boxes.

    Slow and obviously correct: it walks the four deck edges and takes the distance
    from each sample to each box. It converges on the true gap from above as `steps`
    rises, so it is an upper bound on the exact distance and a lower bound on the error
    of anything that reports more than it does.
    """
    px, py = _deck_corners(pos, yaw)
    best = torch.full((pos.shape[0],), 1e3)
    for i in range(4):
        ax, ay = px[:, i], py[:, i]
        bx, by = px[:, (i + 1) % 4], py[:, (i + 1) % 4]
        for k in range(steps + 1):
            u = k / steps
            qx, qy = ax + u * (bx - ax), ay + u * (by - ay)
            for cx, cy, hx, hy in BOXES:
                dx, dy = (qx - cx).abs() - hx, (qy - cy).abs() - hy
                out = torch.sqrt(torch.clamp(dx, min=0.0) ** 2
                                 + torch.clamp(dy, min=0.0) ** 2)
                ins = torch.clamp(torch.maximum(dx, dy), max=0.0)
                best = torch.minimum(best, out + ins)
    return best


# (name, deck centre x, y, yaw, expected gap)
CASES = [
    ('slot centre, square',        -0.80, 0.00, 0.0, 0.5 - DECK_HALF_W),
    ('square, 10 mm off the wall', -0.80, 0.21, 0.0, 0.01),
    ('beside the bay, square',     -0.80, 1.00, 0.0, 1.00 - DECK_HALF_W - 0.60),
    ('deep, near the back wall',   -1.00, 0.00, 0.0, D - 1.00 - DECK_HALF_L),
    ('in the mouth, turned 90 deg', 0.00, 0.00, math.pi / 2, 0.5 - DECK_HALF_L),
]

print(f'{"case":32s} {"gap":>8s} {"expect":>8s}')
bad = 0
for name, x, y, yaw, want in CASES:
    pos = torch.tensor([[x, y]])
    px, py = _deck_corners(pos, torch.tensor([yaw]))
    got = _bay_gap(px, py, Cfg).item()
    ok = abs(got - want) < 1e-5
    bad += not ok
    print(f'{name:32s} {got:8.4f} {want:8.4f}  {"ok" if ok else "FAIL"}')

# The case the corner-only form gets wrong: the deck lying across the front corner of
# a rack, grazing it, with every deck corner far outside the wall box. Found by search
# over disjoint poses, so the exact value is a true distance and not a penetration.
pos = torch.tensor([[0.2828, 0.5373]])
yaw = torch.tensor([-1.62682])
px, py = _deck_corners(pos, yaw)
exact = _bay_gap(px, py, Cfg).item()
corner = corner_only_gap(px, py).item()
ref = reference(pos, yaw).item()
print(f'\ndeck grazing the rack corner: exact {exact*1000:7.2f} mm, '
      f'corner-only {corner*1000:8.2f} mm, reference {ref*1000:7.2f} mm')
if not (exact < 0.01 <= corner - 0.3):
    print('FAIL: this case was meant to show a contact reported as 300+ mm of clearance')
    bad += 1
if abs(exact - ref) > 2e-3:
    print('FAIL: disagrees with the sampled reference')
    bad += 1

# Random sweep against the reference, over poses the colliders allow.
torch.manual_seed(0)
n = 4000
pos = torch.stack([torch.rand(n) * 3.5 - 2.0, torch.rand(n) * 3.2 - 1.6], dim=1)
yaw = torch.rand(n) * 2 * math.pi - math.pi
px, py = _deck_corners(pos, yaw)
new = _bay_gap(px, py, Cfg)
old = corner_only_gap(px, py)
ref = reference(pos, yaw, steps=200)
keep = ref >= 0.0                       # no interpenetration: what PhysX permits
err = (new[keep] - ref[keep]).abs()
print(f'\n{int(keep.sum())} reachable poses: max |exact - reference| '
      f'{err.max().item()*1000:.2f} mm')
if err.max().item() > 3e-3:
    print('FAIL: the exact form disagrees with the sampled reference')
    bad += 1
missed = (old[keep] >= 0.01) & (ref[keep] < 0.01)
print(f'corner-only called {int(missed.sum())} of them clear while the deck was '
      f'touching ({100*missed.float().mean().item():.2f}%)')

print('\nFAILURES:', bad)
sys.exit(1 if bad else 0)
