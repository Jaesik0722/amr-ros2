"""Build the robot-plus-cart USD that the learning task uses.

The ROS simulator (``amr_isaac/``) keeps the robot and the cart as two articulations
joined by a latch joint that is enabled when the hook catches, because there the point
is the docking manoeuvre. Here the cart is coupled for the whole episode, so robot and
cart are built as **one articulation** instead. That matters for two reasons:

- a joint between two articulations cannot be created while simulating (and thousands of
  them would have to be, one per environment), and
- as one articulation the hitch yaw is a real joint of the tree, so PhysX reports its
  measured effort and accepts a position target on it. That joint *is* the hitch angle
  beta: the hook holds the tow bar in a bearing that turns about Y, so the only yaw
  freedom between robot and cart is the hitch yaw servo. Towing with the servo limp is
  the passive trailer this repo has measured so far; driving it is the active hitch the
  task is about.

Geometry comes from the same places as the simulator - the generated inertials for the
robot, ``docking_targets.yaml`` for the coupling, ``world.py`` for the cart - so the two
cannot drift apart. The visual meshes are left out: collision primitives carry the same
inertia, and 4096 copies of the CAD meshes do not fit in the GPU.
"""

import math
import os

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

# ---- robot, from amr.urdf.xacro and the generated amr_inertials.xacro ----------------
BASE_Z = 0.100                      # base_link above base_footprint [m]
WHEEL_RADIUS = 0.100
WHEEL_Y = 0.1871                    # half of the 0.3742 m track
TYRE_CONTACT_WIDTH = 0.012          # see CLAUDE.md pitfall 6
YAW_IN_BASE = (-0.1768, 0.0, -0.0052)
PITCH_IN_YAW = (-0.0422, 0.0, 0.0)
TIP_IN_PITCH = (-0.0496, 0.0, 0.0200)
ROBOT_CASTERS = ((0.127, 0.0, -0.080), (-0.127, 0.0, -0.080))
CASTER_RADIUS = 0.020
# The CAD has the support casters exactly flush with the tyres, so all four contacts are
# coplanar and the load split between them is indeterminate. The casters are modelled as
# frictionless, so whenever the solver puts weight on them the drive wheels lose the
# normal force they need and the robot stays put with its wheels spinning. At 3 mm of
# clearance that still happened on about half the runs, at any payload (0.397 +- 0.162
# m/s against a 0.5 m/s command at zero payload); at 10 mm it does not (0.497 +- 0.005
# at 100 kg).
#
# This is a modelling choice, not a measurement. A real build cannot have the casters
# flush either - the drive wheels would have no traction - so there is clearance on the
# robot, and a 200 mm tyre compresses a few mm under 20 kg. How much should be measured
# on the hardware.
CASTER_CLEARANCE = 0.010
# The real build does not lift both of them. The front caster went in about 10 mm up, the
# same as this model, and the rear one is being assembled flush with the tyres (CLAUDE.md
# pitfall 33). Flush at the rear is not the coplanar case the comment above warns about:
# with the front caster still clear, the robot stands on two drive wheels and one rear
# caster, which is a determinate three-point support. What it does do is hand part of the
# load to a frictionless sphere, so the drive wheels carry less than this model gives
# them. None means "whatever CASTER_CLEARANCE is".
REAR_CASTER_CLEARANCE = None

BASE_MASS = 12.4805
BASE_COM = (-0.00941, -0.00009, 0.04855)
BASE_INERTIA = (1.294830e-01, 1.497727e-01, 1.431309e-01)
BASE_BOXES = (((0.267, 0.274, 0.218), (-0.0015, 0.0, 0.073)),
              ((0.067, 0.270, 0.006), (-0.1685, 0.0, -0.033)),
              ((0.194, 0.220, 0.047), (-0.018, 0.0, 0.204)),
              ((0.063, 0.136, 0.136), (0.1625, 0.0, 0.017)))
WHEEL_MASS = 3.8000
WHEEL_INERTIA = (1.169836e-02, 2.111097e-02, 1.169821e-02)
YAW_MASS, YAW_COM = 0.2927, (-0.01712, -0.00041, 0.00199)
YAW_INERTIA = (1.065805e-04, 2.348790e-04, 2.463903e-04)
PITCH_MASS, PITCH_COM = 0.0503, (-0.02318, 0.0, 0.00868)
PITCH_INERTIA = (3.557042e-05, 2.901505e-05, 5.322478e-05)

# ---- cart, from world.py and docking_targets.yaml ------------------------------------
TOW_BAR_IN_CART = (0.58, 0.0, 0.115)
MARKER_IN_CART_X = 0.43             # target_in_cart, where the docking plate sits
DECK_HALF_L, DECK_HALF_W = 0.42, 0.28
DECK_TOP, DECK_THICK = 0.26, 0.04
CART_WHEEL_RADIUS = 0.0625
CART_WHEEL_MASS, CART_SWIVEL_MASS = 0.5, 0.3
CART_CASTER_TRAIL = 0.035
CART_WHEEL_X = DECK_HALF_L - 0.08
CART_WHEEL_Y = DECK_HALF_W - 0.06
# The deck itself with its three boxes: the mass the campaign measured, before payload.
CART_BASE_MASS = 22.6
CART_BASE_COM = (-0.03, 0.0, 0.30)
# Load box the payload is added inside, and the empty cart's bounding box for inertia.
CART_BOX = (2 * DECK_HALF_L, 2 * DECK_HALF_W, 0.52)
PAYLOAD_BOX = (0.50, 0.40, 0.30)

# Hitch pitch target while towing (``hook_up`` in amr_docking/commands.py): the hook is
# held up under the bar, which is what loads the pitch servo.
PITCH_TOW = 0.2

ROOT = '/amr_tow'


def _xform(stage, path, pos=(0.0, 0.0, 0.0), rot_z=0.0):
    prim = stage.DefinePrim(path, 'Xform')
    x = UsdGeom.Xformable(prim)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot_z:
        x.AddRotateZOp().Set(math.degrees(rot_z))
    return prim


def _body(stage, path, mass, com, inertia, pos=(0.0, 0.0, 0.0)):
    prim = _xform(stage, path, pos)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    return prim


def _box(stage, path, size, pos, material=None, bind=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if material is not None:
        bind(cube.GetPrim(), material)
    return cube.GetPrim()


def _cylinder(stage, path, radius, length, pos, axis='Y', material=None, bind=None):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    cyl.CreateAxisAttr(axis)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    if material is not None:
        bind(cyl.GetPrim(), material)
    return cyl.GetPrim()


def _sphere(stage, path, radius, pos, material=None, bind=None):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    if material is not None:
        bind(sph.GetPrim(), material)
    return sph.GetPrim()


# The CAD robot the ROS simulator imports from the URDF. Five of its links carry meshes,
# and each sits at its link origin with no transform of its own, so referencing one under
# the matching body here reproduces that view exactly.
# The reference is to /visuals/<link>, not /meshes/<link>: the latter is the bare
# geometry and renders white, while /visuals carries the CATIA materials bound through
# per-face GeomSubsets, which is what makes the robot look like the robot.
CAD_USD = os.path.expanduser('~/.cache/amr_isaac/usd/configuration/amr_base.usd')
CAD_LINKS = ('base_link', 'hitch_yaw_link', 'hitch_pitch_link',
             'left_wheel_link', 'right_wheel_link')


def _attach_visual(stage, body_path, link):
    """Put the link's CAD mesh under body_path as visual-only geometry."""
    prim = stage.DefinePrim(f'{body_path}/visuals', 'Xform')
    prim.GetReferences().AddReference(CAD_USD, f'/visuals/{link}')
    # No CollisionAPI is applied: the collision primitives built above stay the physics,
    # and a 2.4 M-point hull would make the solver useless.
    return prim


def _linearize_cad_colours(stage, root):
    """Convert the CAD shader colours from sRGB to linear, once.

    The CATIA meshes carry their colours as sRGB values, which is what RViz displays
    directly, but Omniverse treats a material colour as linear and brightens it on
    output --- so the robot's dark grey renders light grey. ``amr_isaac/amr_sim.py``
    does the same conversion when it imports the URDF; this is that step, applied while
    the asset is baked rather than at load time.
    """
    def to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Shader' or \
                not prim.GetPath().pathString.startswith(root):
            continue
        attr = prim.GetAttribute('inputs:diffuse_color_constant')
        if attr and attr.Get() is not None and \
                not prim.GetCustomDataByKey('amr:linearized'):
            attr.Set(Gf.Vec3f(*(to_linear(c) for c in attr.Get())))
            prim.SetCustomDataByKey('amr:linearized', True)


# The cart's own collision shapes that the dressing replaces. The deck slab and the tow
# bar are kept: they are the right size in the right place, so they are the visual too.
CART_HULLS_HIDDEN = ('/load', '/tyre', '/front_support_')


def _hide_collision_shapes(stage, root):
    """Make the collision primitives invisible, leaving the visual geometry to show."""
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(root) or '/visuals' in path or '/v_' in path:
            continue
        if not (prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder) or
                prim.IsA(UsdGeom.Sphere)):
            continue
        if '/cart_' in path and not any(h in path for h in CART_HULLS_HIDDEN):
            continue
        UsdGeom.Imageable(prim).MakeInvisible()


def _srgb(r, g, b):
    """8-bit sRGB colour -> linear displayColor, as ``amr_isaac/world.py`` does it."""
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return Gf.Vec3f(lin(r), lin(g), lin(b))


# The cart has no CAD -- it is a generic folding platform cart -- so, exactly as in
# ``amr_isaac/world.py``, it is dressed with primitives: a green plastic deck with a
# raised rim and ribs, the white tube handle at the far end, the cardboard load, and
# plate-fork-wheel casters. The physics shapes built above stay the physics; everything
# here is visual only, which is why it can be a hundred small prims. Without it the cart
# renders as its hulls -- one slab, one block and four balls where the wheels are -- and
# that is not the cart the ROS recordings show.
GREEN, LIP = (62, 128, 58), (52, 112, 48)
WHITE, METAL, DARK = (236, 234, 222), (165, 168, 172), (25, 25, 25)
TYRE_GREY, CARDBOARD, TAPE = (110, 112, 115), (196, 150, 98), (214, 190, 150)
HANDLE_HEIGHT = 0.86
# Cardboard boxes as world.py stacks them: two on the deck, a third on top of the front
# one. The collision hull is the single ``load`` box, which is hidden when these appear.
CART_LOAD_BOXES = (((0.40, 0.48, 0.30), (0.17, 0.0, 0.0)),
                   ((0.34, 0.44, 0.26), (-0.20, 0.0, 0.0)),
                   ((0.30, 0.36, 0.22), (0.15, 0.02, 0.30)))


def _vbox(stage, path, size, pos, rgb):
    """Visual-only box: no collider, no mass, just a coloured slab."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([_srgb(*rgb)])
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    return cube.GetPrim()


def _vtube(stage, path, radius, length, pos, rgb, axis='Y'):
    """Visual-only cylinder: a handle tube, a hinge rod, a tyre."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    cyl.CreateAxisAttr(axis)
    cyl.CreateDisplayColorAttr([_srgb(*rgb)])
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*pos))
    return cyl.GetPrim()


def _dress_cart(stage, wheels):
    """Hang world.py's cart appearance on the bodies this asset already built."""
    b = f'{ROOT}/cart_body'
    hl, hw, top, thick = DECK_HALF_L, DECK_HALF_W, DECK_TOP, DECK_THICK
    r = CART_WHEEL_RADIUS
    # Deck: the collision slab is the deck, so it only needs its colour. The rim and the
    # ribs are mouldings a few millimetres proud of it.
    deck = stage.GetPrimAtPath(f'{b}/deck')
    if deck:
        UsdGeom.Gprim(deck).CreateDisplayColorAttr([_srgb(*GREEN)])
    for name, size, pos in (
            ('rim_f', (0.02, 2 * hw, 0.012), (hl - 0.01, 0.0, top + 0.006)),
            ('rim_b', (0.02, 2 * hw, 0.012), (-hl + 0.01, 0.0, top + 0.006)),
            ('rim_l', (2 * hl, 0.02, 0.012), (0.0, hw - 0.01, top + 0.006)),
            ('rim_r', (2 * hl, 0.02, 0.012), (0.0, -hw + 0.01, top + 0.006))):
        _vbox(stage, f'{b}/v_{name}', size, pos, LIP)
    for i, y in enumerate((-0.18, -0.06, 0.06, 0.18)):
        _vbox(stage, f'{b}/v_rib{i}', (2 * hl - 0.08, 0.012, 0.003),
              (0.0, y, top + 0.0015), LIP)
    # Marker plate and tow tongue under the front edge: the hardware the docking paper
    # looks at, and the reason the bar sits where it does.
    tx = MARKER_IN_CART_X
    _vbox(stage, f'{b}/v_marker_plate', (0.012, 0.52, 0.125),
          (tx - 0.006, 0.0, top - 0.0625), METAL)
    bx, _, bz = TOW_BAR_IN_CART
    _vbox(stage, f'{b}/v_tongue', (bx - tx + 0.02, 0.06, 0.01),
          ((bx + tx) / 2, 0.0, bz + 0.03), DARK)
    for name, side in (('l', 1), ('r', -1)):
        _vbox(stage, f'{b}/v_tongue_drop_{name}', (0.02, 0.006, 0.035),
              (bx, side * 0.027, bz + 0.012), DARK)
    bar = stage.GetPrimAtPath(f'{b}/tow_bar')
    if bar:
        UsdGeom.Gprim(bar).CreateDisplayColorAttr([_srgb(*WHITE)])
    # Folding handle: two posts on hinges with three cross bars. It is the tallest part
    # of the cart and the thing that makes it read as a platform cart at a glance.
    hx, tube = -hl + 0.03, 0.025
    for name, y in (('l', hw - 0.04), ('r', -hw + 0.04)):
        _vbox(stage, f'{b}/v_hinge_{name}', (0.07, 0.012, 0.07),
              (hx, y + math.copysign(0.02, y), top + 0.03), METAL)
        _vtube(stage, f'{b}/v_handle_post_{name}', tube, HANDLE_HEIGHT,
               (hx, y, top + HANDLE_HEIGHT / 2), WHITE, axis='Z')
    for i, z in enumerate((top + HANDLE_HEIGHT, top + 0.62, top + 0.38)):
        _vtube(stage, f'{b}/v_handle_bar{i}', tube, 2 * hw - 0.08, (hx, 0.0, z), WHITE)
    _vtube(stage, f'{b}/v_hinge_rod', 0.012, 2 * hw - 0.04, (hx, 0.0, top + 0.03), METAL)
    # Load: the three boxes with a tape stripe, in place of the one collision block.
    for i, (size, pos) in enumerate(CART_LOAD_BOXES):
        z = top + pos[2] + size[2] / 2
        _vbox(stage, f'{b}/v_box{i}', size, (pos[0], pos[1], z), CARDBOARD)
        _vbox(stage, f'{b}/v_box{i}_tape', (0.06, size[1] + 0.002, size[2] + 0.002),
              (pos[0], pos[1], z), TAPE)
    # Casters. The mounting plate is bolted to the deck in both carts. Where the caster
    # is idealised (the trailer's front support, a sphere rigid with the deck) the swivel
    # and fork belong on the deck too; where it is the mechanism they have to turn with
    # it, so they go on the swivel body and the tyre on the wheel body.
    for name, x, y in (('fl', CART_WHEEL_X, CART_WHEEL_Y),
                       ('fr', CART_WHEEL_X, -CART_WHEEL_Y),
                       ('bl', -CART_WHEEL_X, CART_WHEEL_Y),
                       ('br', -CART_WHEEL_X, -CART_WHEEL_Y)):
        _vbox(stage, f'{b}/v_caster_plate_{name}', (0.10, 0.08, 0.006),
              (x, y, top - thick - 0.003), METAL)
    if wheels == 'caster4':
        for name in ('fl', 'fr', 'bl', 'br'):
            swivel = f'{ROOT}/cart_swivel_{name}'
            for k, dy in (('a', 0.022), ('b', -0.022)):
                _vbox(stage, f'{swivel}/v_fork{k}', (0.03, 0.005, 0.07),
                      (-CART_CASTER_TRAIL / 2, dy, 0.045), METAL)
            _vbox(stage, f'{swivel}/v_yoke', (0.05, 0.05, 0.02),
                  (0.0, 0.0, 0.085), METAL)
            _vtube(stage, f'{ROOT}/cart_wheel_{name}/v_tyre', r, 0.032,
                   (0.0, 0.0, 0.0), TYRE_GREY)
    else:
        for side, sy in (('l', 1.0), ('r', -1.0)):
            y = sy * CART_WHEEL_Y
            # Front support: idealised, so the whole caster is drawn on the deck.
            _vbox(stage, f'{b}/v_caster_swivel_{side}', (0.05, 0.05, 0.02),
                  (CART_WHEEL_X, y, top - thick - 0.016), METAL)
            for k, dy in (('a', 0.022), ('b', -0.022)):
                _vbox(stage, f'{b}/v_caster_fork_{side}{k}',
                      (0.03, 0.005, top - thick - r - 0.026),
                      (CART_WHEEL_X, y + dy, (top - thick - 0.026 + r) / 2), METAL)
            _vtube(stage, f'{b}/v_caster_wheel_{side}', r, 0.032,
                   (CART_WHEEL_X, y, r), TYRE_GREY)
            _vtube(stage, f'{ROOT}/cart_wheel_b{side}/v_tyre', r, 0.032,
                   (0.0, 0.0, 0.0), TYRE_GREY)


def _material(stage, path, static, dynamic, combine):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    physx.CreateFrictionCombineModeAttr(combine)
    physx.CreateRestitutionCombineModeAttr('min')
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, 'physics')


def _revolute(stage, path, body0, body1, axis, pos0, pos1, limits=None, drive=None):
    """Define one revolute joint. `limits` in radians, `drive` as (stiffness, damping)."""
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    j.CreateCollisionEnabledAttr(False)
    if limits is not None:                       # USD joint limits are in degrees
        j.CreateLowerLimitAttr(math.degrees(limits[0]))
        j.CreateUpperLimitAttr(math.degrees(limits[1]))
    # Every joint carries a drive so Isaac Lab's actuator config has something to write
    # to; the free ones (axles, swivels, the bar in the hook) get stiffness 0 and only
    # the viscous damping that stands for bearing friction.
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'angular')
    stiffness, damping = drive if drive is not None else (0.0, 0.0)
    d.CreateTypeAttr('force')                    # see CLAUDE.md pitfall 3
    d.CreateTargetPositionAttr(0.0)
    d.CreateStiffnessAttr(stiffness)
    d.CreateDampingAttr(damping)
    return j


def cart_inertia(mass, box=CART_BOX):
    """Diagonal inertia of a uniform box of this mass, about its own centre."""
    sx, sy, sz = box
    return (mass / 12.0 * (sy * sy + sz * sz),
            mass / 12.0 * (sx * sx + sz * sz),
            mass / 12.0 * (sx * sx + sy * sy))


def build(stage, wheels='trailer', visuals=False):
    """Write the one-articulation robot+cart into `stage` and return its root path."""
    if wheels not in ('trailer', 'caster4'):
        raise ValueError(f'unknown cart wheels {wheels!r}: trailer or caster4')

    # Combine mode 'min' throughout, so a shape's own coefficient is what acts against
    # the floor. The ROS simulator uses 'max' for the tyre, which is harmless there, but
    # here the floor friction is randomised per environment: with 'max' against a 1.0
    # ground plane every tyre would silently come out at 1.0 whatever it was set to.
    tyre = _material(stage, f'{ROOT}/materials/tyre', 1.0, 0.9, 'min')
    free = _material(stage, f'{ROOT}/materials/frictionless', 0.0, 0.0, 'min')
    cart_tyre = _material(stage, f'{ROOT}/materials/cart_tyre', 0.9, 0.8, 'min')

    root = _xform(stage, ROOT)
    stage.SetDefaultPrim(root)

    # ---- robot ----------------------------------------------------------------------
    base = _body(stage, f'{ROOT}/base_link', BASE_MASS, BASE_COM, BASE_INERTIA,
                 pos=(0.0, 0.0, BASE_Z))
    # The articulation root goes on base_link, not on the enclosing Xform. With it on the
    # Xform, PhysX picks a root link itself - it picked hitch_yaw_link - and every root
    # pose written at reset then placed the hitch instead of the robot, leaving the robot
    # 176.8 mm (the hitch offset) ahead of where the task thought it was.
    UsdPhysics.ArticulationRootAPI.Apply(base)
    PhysxSchema.PhysxArticulationAPI.Apply(base)
    for i, (size, pos) in enumerate(BASE_BOXES):
        _box(stage, f'{ROOT}/base_link/hull{i}', size, pos)
    # The two support casters are idealised as frictionless spheres rigid with the body,
    # as in the simulator: they hold the chassis up and add no steering resistance.
    for i, (cx, cy, cz) in enumerate(ROBOT_CASTERS):
        lift = CASTER_CLEARANCE
        if cx < 0.0 and REAR_CASTER_CLEARANCE is not None:
            lift = REAR_CASTER_CLEARANCE
        _sphere(stage, f'{ROOT}/base_link/caster{i}', CASTER_RADIUS,
                (cx, cy, cz + lift), free, _bind)

    for name, sign in (('left', 1.0), ('right', -1.0)):
        link = f'{ROOT}/{name}_wheel_link'
        _body(stage, link, WHEEL_MASS, (0.0, 0.0, 0.0), WHEEL_INERTIA,
              pos=(0.0, sign * WHEEL_Y, BASE_Z))
        # Narrow tread: the full 46 mm tyre width turns the contact into a scrub patch
        # and the robot only reaches 90% of its commanded yaw rate (CLAUDE.md pitfall 6).
        _cylinder(stage, f'{link}/tread', WHEEL_RADIUS, TYRE_CONTACT_WIDTH,
                  (0.0, 0.0, 0.0), 'Y', tyre, _bind)
        _revolute(stage, f'{ROOT}/joints/{name}_wheel_joint', base.GetPath(), link, 'Y',
                  (0.0, sign * WHEEL_Y, 0.0), (0.0, 0.0, 0.0))

    yaw_link = f'{ROOT}/hitch_yaw_link'
    _body(stage, yaw_link, YAW_MASS, YAW_COM, YAW_INERTIA,
          pos=(YAW_IN_BASE[0], 0.0, BASE_Z + YAW_IN_BASE[2]))
    _box(stage, f'{yaw_link}/hull', (0.091, 0.048, 0.038), (-0.0127, 0.0, 0.0))
    _revolute(stage, f'{ROOT}/joints/hitch_yaw_joint', base.GetPath(), yaw_link, 'Z',
              YAW_IN_BASE, (0.0, 0.0, 0.0), (-math.pi / 2, math.pi / 2))

    pitch_link = f'{ROOT}/hitch_pitch_link'
    _body(stage, pitch_link, PITCH_MASS, PITCH_COM, PITCH_INERTIA,
          pos=(YAW_IN_BASE[0] + PITCH_IN_YAW[0], 0.0, BASE_Z + YAW_IN_BASE[2]))
    _box(stage, f'{pitch_link}/hook', (0.070, 0.030, 0.030),
         (TIP_IN_PITCH[0] / 2, 0.0, TIP_IN_PITCH[2] / 2))
    _revolute(stage, f'{ROOT}/joints/hitch_pitch_joint', yaw_link, pitch_link, 'Y',
              PITCH_IN_YAW, (0.0, 0.0, 0.0), (-math.pi / 2, math.pi / 2))

    # ---- cart -----------------------------------------------------------------------
    # Cart frame: centre on the floor, x towards the tow bar. Placed so the bar sits in
    # the hook with the pitch joint at zero; the pitch target then lifts it as on the
    # real robot.
    tip_x = YAW_IN_BASE[0] + PITCH_IN_YAW[0] + TIP_IN_PITCH[0]
    tip_z = BASE_Z + YAW_IN_BASE[2] + TIP_IN_PITCH[2]
    cart_x = tip_x - TOW_BAR_IN_CART[0]
    cart_z = tip_z - TOW_BAR_IN_CART[2]
    body = f'{ROOT}/cart_body'
    parts = (2 * CART_WHEEL_MASS if wheels == 'trailer'
             else 4 * (CART_SWIVEL_MASS + CART_WHEEL_MASS))
    deck_mass = CART_BASE_MASS - parts
    _body(stage, body, deck_mass, CART_BASE_COM, cart_inertia(deck_mass),
          pos=(cart_x, 0.0, cart_z))
    _box(stage, f'{body}/deck', (2 * DECK_HALF_L, 2 * DECK_HALF_W, DECK_THICK),
         (0.0, 0.0, DECK_TOP - DECK_THICK / 2))
    _box(stage, f'{body}/load', (0.74, 0.50, 0.32), (-0.02, 0.0, DECK_TOP + 0.16))
    _cylinder(stage, f'{body}/tow_bar', 0.006, 0.06, TOW_BAR_IN_CART, 'Y')
    # The bar turns freely in the hook, about the bar axis only. So the hitch yaw joint
    # is the single yaw freedom between robot and cart - it is beta.
    _revolute(stage, f'{ROOT}/joints/tow_bar_joint', pitch_link, body, 'Y',
              TIP_IN_PITCH, TOW_BAR_IN_CART, drive=(0.0, 0.02))

    r = CART_WHEEL_RADIUS
    for side, sy in (('l', 1.0), ('r', -1.0)):
        y = sy * CART_WHEEL_Y
        if wheels == 'trailer':
            # Front: frictionless spheres rigid with the deck, so the only ground
            # constraint is the rear axle and the cart has a heading. Textbook trailer.
            _sphere(stage, f'{body}/front_support_{side}', r, (CART_WHEEL_X, y, r), free, _bind)
            wheel = f'{ROOT}/cart_wheel_b{side}'
            _body(stage, wheel, CART_WHEEL_MASS, (0.0, 0.0, 0.0),
                  cart_inertia(CART_WHEEL_MASS, (r, r, r)),
                  pos=(cart_x - CART_WHEEL_X, y, r))
            _sphere(stage, f'{wheel}/tyre', r, (0.0, 0.0, 0.0), cart_tyre, _bind)
            _revolute(stage, f'{ROOT}/joints/cart_axle_b{side}', body, wheel, 'Y',
                      (-CART_WHEEL_X, y, r), (0.0, 0.0, 0.0), drive=(0.0, 0.02))
            continue
        # Four swivelling casters, each built as the mechanism: a swivel about the
        # vertical mount axis carrying a wheel offset behind it by the trail. The trail
        # is what lines a caster up with the motion and resists being dragged sideways.
        for place, sx in (('f', 1.0), ('b', -1.0)):
            x = sx * CART_WHEEL_X
            name = f'{place}{side}'
            swivel = f'{ROOT}/cart_swivel_{name}'
            _body(stage, swivel, CART_SWIVEL_MASS, (0.0, 0.0, 0.0),
                  cart_inertia(CART_SWIVEL_MASS, (0.05, 0.05, 0.07)),
                  pos=(cart_x + x, y, r))
            _revolute(stage, f'{ROOT}/joints/cart_swivel_{name}', body, swivel, 'Z',
                      (x, y, r), (0.0, 0.0, 0.0), drive=(0.0, 0.001))
            wheel = f'{ROOT}/cart_wheel_{name}'
            _body(stage, wheel, CART_WHEEL_MASS, (0.0, 0.0, 0.0),
                  cart_inertia(CART_WHEEL_MASS, (r, r, r)),
                  pos=(cart_x + x - CART_CASTER_TRAIL, y, r))
            _sphere(stage, f'{wheel}/tyre', r, (0.0, 0.0, 0.0), cart_tyre, _bind)
            _revolute(stage, f'{ROOT}/joints/cart_axle_{name}', swivel, wheel, 'Y',
                      (-CART_CASTER_TRAIL, 0.0, 0.0), (0.0, 0.0, 0.0), drive=(0.0, 0.02))

    if visuals:
        # For a figure or a recording, where there is one environment rather than 4096.
        # The physics is untouched: the meshes carry no collider and no mass, they only
        # replace what the camera sees.
        for link in CAD_LINKS:
            _attach_visual(stage, f'{ROOT}/{link}', link)
        _linearize_cad_colours(stage, ROOT)
        _dress_cart(stage, wheels)
        _hide_collision_shapes(stage, ROOT)
    return ROOT


def usd_path(wheels, cache=None, visuals=False, clearance=None, rear=None):
    cache = cache or os.path.expanduser('~/.cache/amr_lab')
    suffix = '_visual' if visuals else ''
    if clearance is not None:
        suffix += f'_c{round(clearance * 1000)}mm'
    if rear is not None:
        suffix += f'_r{round(rear * 1000)}mm'
    return os.path.join(cache, f'amr_tow_{wheels}{suffix}.usd')


def ensure(wheels='trailer', cache=None, rebuild=False, visuals=False, clearance=None,
           rear_clearance=None):
    """Build the asset if it is not cached, and return the path to it.

    ``visuals`` builds a second, separately cached asset that carries the CAD meshes
    for rendering. It is not the training asset: the meshes do not fit 4096 times.

    ``clearance`` overrides ``CASTER_CLEARANCE`` and caches the result separately. The
    10 mm default is a modelling choice rather than a measurement (see the comment on
    the constant), so how much of the result depends on it is a question the paper has
    to answer rather than declare, and answering it needs an asset built at another
    value. The constant is set for the duration of the build because that is where it is
    read; nothing else in the module keeps a copy.

    ``rear_clearance`` does the same for the rear caster alone, which is how the robot is
    actually being assembled: front clear, rear flush. A checkpoint trained with both
    clear and replayed on that asset is an out-of-distribution check, not a retrained
    result.
    """
    if visuals and not os.path.exists(CAD_USD):
        raise FileNotFoundError(
            f'{CAD_USD} is missing. It is written the first time amr_isaac imports the '
            'URDF, so run ./amr_isaac/run_sim.sh --headless --selftest once.')
    path = usd_path(wheels, cache, visuals, clearance, rear_clearance)
    if os.path.exists(path) and not rebuild and \
            os.path.getmtime(path) > os.path.getmtime(__file__):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    global CASTER_CLEARANCE, REAR_CASTER_CLEARANCE
    saved = (CASTER_CLEARANCE, REAR_CASTER_CLEARANCE)
    if clearance is not None:
        CASTER_CLEARANCE = clearance
    if rear_clearance is not None:
        REAR_CASTER_CLEARANCE = rear_clearance
    try:
        build(stage, wheels, visuals)
    finally:
        CASTER_CLEARANCE, REAR_CASTER_CLEARANCE = saved
    stage.GetRootLayer().Export(path)
    return path
