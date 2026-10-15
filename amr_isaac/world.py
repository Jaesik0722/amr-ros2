"""Cart, charging station, ArUco markers, indoor GPS, battery and cart latch.

Geometry that the robot's software relies on (marker ids/positions, tow bar, docked
poses) comes from amr_docking/config/docking_targets.yaml, so the simulator and the
detector cannot drift apart. The rest of the cart (a folding platform cart with boxes)
and charger shape is illustrative.

ROS interface (same as on the real robot where noted):
  /indoor_gps/cart  PoseStamped, map frame, 5 Hz, 2 cm / 1 deg noise (Marvelmind-like)
  /battery_state    BatteryState, current > 0 while charging (opennav_docking)
  /cart/latched     Bool (simulator ground truth; the robot checks the hitch current)

Coupling works like the real hook: with the hook within 3 cm (horizontally) of the bar,
raising the hitch pitch above ENGAGE_PITCH catches the bar (a revolute joint about the
bar axis is created); lowering it below RELEASE_PITCH lets go. /cart/latch (SetBool)
forces it for manual tests.
"""

import math
import os

from geometry_msgs.msg import PoseStamped
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
import yaml

CART = '/World/cart'
CHARGER = '/World/charger'
LATCH_JOINT = '/World/cart_latch'
# Hitch kinematics (amr.urdf.xacro): base_footprint -> yaw axis -> pitch axis -> hook tip.
YAW_IN_BASE = np.array([-0.1768, 0.0, 0.1 - 0.0052])
PITCH_IN_YAW = np.array([-0.0422, 0.0, 0.0])
TIP_IN_PITCH = np.array([-0.0496, 0.0, 0.0200])
RX_IN_BASE = np.array([0.194, 0.0, 0.117])     # charging_rx_link
TX_IN_CHARGER = np.array([0.0, 0.0, 0.117])     # TC-200 antenna face, charger target frame
LATCH_TOLERANCE = 0.03   # m, horizontal hook-to-bar offset that still catches
ENGAGE_PITCH = 0.1     # rad, hitch pitch target that catches the bar
RELEASE_PITCH = 0.0


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def place(prim, xyz, yaw=0.0):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))


def box(stage, path, size, pos, color, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def srgb(r, g, b):
    """8-bit sRGB colour -> linear displayColor (what the renderer expects)."""
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b))


def tube_y(stage, path, radius, length, pos, color, axis='Y'):
    """Visual-only cylinder (tube, wheel) along `axis`."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    cyl.CreateAxisAttr(axis)
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*pos))
    return cyl.GetPrim()


def marker_quad(stage, path, png, size, pos):
    """Textured square facing +x of its parent, with the image upright seen from +x."""
    h = size / 2.0
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(0, -h, -h), Gf.Vec3f(0, h, -h),
                           Gf.Vec3f(0, h, h), Gf.Vec3f(0, -h, h)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(1, 0, 0)] * 4)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*pos))
    mat = UsdShade.Material.Define(stage, path + '_mat')
    shader = UsdShade.Shader.Define(stage, path + '_mat/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.9)
    reader = UsdShade.Shader.Define(stage, path + '_mat/st')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    tex = UsdShade.Shader.Define(stage, path + '_mat/texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(png)
    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), 'result')
    tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), 'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


class DockingWorld:
    """Build the cart and charger; simulate the indoor GPS, battery and latch."""

    def __init__(self, stage, targets_file, markers_dir, cart_pose, charger_pose,
                 friction_materials, bind_material):
        with open(targets_file) as f:
            self.cfg = yaml.safe_load(f)
        self.stage = stage
        self.markers_dir = markers_dir
        self.cart_pose = cart_pose
        self.charger_pose = charger_pose
        self.marker_size = self.cfg['aruco']['marker_size'] * 8.0 / 6.0  # incl. quiet zone
        self.tyre_mat, self.caster_mat = friction_materials
        self.bind_material = bind_material
        self.cart = None
        self.latched = False
        self.latch_request = None
        self.rng = np.random.default_rng(0)
        self.battery = 0.8
        self.charging = False
        if cart_pose is not None:
            self._build_cart()
        if charger_pose is not None:
            self._build_charger()

    # ---- scene ----------------------------------------------------------------
    def _markers(self, parent, target, x):
        t = self.cfg[target]
        for marker_id, y in t['markers'].items():
            png = os.path.join(self.markers_dir, f'aruco_{marker_id}.png')
            marker_quad(self.stage, f'{parent}/aruco_{marker_id}', png, self.marker_size,
                        (x, float(y), t['marker_height']))

    def _build_cart(self, mass=22.0):
        """Folding platform cart (green plastic deck, white handle) loaded with boxes.

        Cart frame: centre on the floor, x towards the robot side (marker plate and tow
        bar), handle at the far end. Deck 0.84 x 0.56 m, top 0.26 m; 125 mm wheels.
        The boxes are part of the cart body (mass includes ~8 kg of load).
        """
        s = self.stage
        root = s.DefinePrim(CART, 'Xform')
        place(root, (self.cart_pose[0], self.cart_pose[1], 0.0), self.cart_pose[2])
        UsdPhysics.ArticulationRootAPI.Apply(root)
        body = s.DefinePrim(f'{CART}/body', 'Xform')
        UsdPhysics.RigidBodyAPI.Apply(body)
        m = UsdPhysics.MassAPI.Apply(body)
        m.CreateMassAttr(mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(-0.03, 0.0, 0.30))
        # Rolling resistance: the casters are ideal (frictionless) and the axles free, so
        # without damping a nudged cart rolls on forever. ~7 N at 0.3 m/s.
        damp = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        damp.CreateLinearDampingAttr(1.0)
        damp.CreateAngularDampingAttr(1.0)
        green, lip = srgb(62, 128, 58), srgb(52, 112, 48)
        white, metal, dark = srgb(236, 234, 222), srgb(165, 168, 172), srgb(25, 25, 25)
        tyre, cardboard, tape = srgb(110, 112, 115), srgb(196, 150, 98), srgb(214, 190, 150)
        b = f'{CART}/body'
        hl, hw, top, thick, r = 0.42, 0.28, 0.26, 0.04, 0.0625
        # Deck: plastic plate with a raised rim and shallow ribs.
        box(s, f'{b}/deck', (2 * hl, 2 * hw, thick), (0.0, 0.0, top - thick / 2), green)
        for name, size, pos in (
                ('rim_f', (0.02, 2 * hw, 0.012), (hl - 0.01, 0.0, top + 0.006)),
                ('rim_b', (0.02, 2 * hw, 0.012), (-hl + 0.01, 0.0, top + 0.006)),
                ('rim_l', (2 * hl, 0.02, 0.012), (0.0, hw - 0.01, top + 0.006)),
                ('rim_r', (2 * hl, 0.02, 0.012), (0.0, -hw + 0.01, top + 0.006))):
            box(s, f'{b}/{name}', size, pos, lip, collide=False)
        for i, y in enumerate((-0.18, -0.06, 0.06, 0.18)):
            box(s, f'{b}/rib{i}', (2 * hl - 0.08, 0.012, 0.003), (0.0, y, top + 0.0015), lip,
                collide=False)
        # Marker plate under the front deck edge (face at x = target_in_cart).
        tx = self.cfg['cart']['target_in_cart'][0]
        box(s, f'{b}/marker_plate', (0.012, 0.52, 0.125), (tx - 0.006, 0.0, top - 0.0625), metal)
        self._markers(b, 'cart', tx + 0.001)
        # Tow tongue and bar under the plate: visual only; the coupling is the latch joint.
        bx, _, bz = self.cfg['cart']['tow_bar_in_cart']
        box(s, f'{b}/tongue', (bx - tx + 0.02, 0.06, 0.01), ((bx + tx) / 2, 0.0, bz + 0.03),
            dark, collide=False)
        for name, side in (('l', 1), ('r', -1)):
            box(s, f'{b}/tongue_drop_{name}', (0.02, 0.006, 0.035),
                (bx, side * 0.027, bz + 0.012), dark, collide=False)
        bar = UsdGeom.Cylinder.Define(s, f'{b}/tow_bar')
        bar.CreateRadiusAttr(0.006)
        bar.CreateHeightAttr(0.06)
        bar.CreateAxisAttr('Y')
        bar.CreateDisplayColorAttr([Gf.Vec3f(*white)])
        UsdGeom.Xformable(bar).AddTranslateOp().Set(Gf.Vec3d(bx, 0.0, bz))
        # Folding handle at the far end: white tube U-frame with two cross bars.
        hx, tube, height = -hl + 0.03, 0.025, 0.86
        for name, y in (('l', hw - 0.04), ('r', -hw + 0.04)):
            box(s, f'{b}/hinge_{name}', (0.07, 0.012, 0.07), (hx, y + math.copysign(0.02, y),
                top + 0.03), metal, collide=False)
            tube_y(s, f'{b}/handle_post_{name}', tube, height, (hx, y, top + height / 2), white,
                   axis='Z')
        for i, z in enumerate((top + height, top + 0.62, top + 0.38)):
            tube_y(s, f'{b}/handle_bar{i}', tube, 2 * hw - 0.08, (hx, 0.0, z), white)
        tube_y(s, f'{b}/hinge_rod', 0.012, 2 * hw - 0.04, (hx, 0.0, top + 0.03), metal)
        # Load: cardboard boxes with a tape stripe.
        for i, (size, pos) in enumerate((((0.40, 0.48, 0.30), (0.17, 0.0, 0.0)),
                                         ((0.34, 0.44, 0.26), (-0.20, 0.0, 0.0)),
                                         ((0.30, 0.36, 0.22), (0.15, 0.02, 0.30)))):
            z = top + pos[2] + size[2] / 2
            box(s, f'{b}/box{i}', size, (pos[0], pos[1], z), cardboard)
            box(s, f'{b}/box{i}_tape', (0.06, size[1] + 0.002, size[2] + 0.002),
                (pos[0], pos[1], z), tape, collide=False)
        # Casters: plate + fork + grey wheel. Physics: frictionless spheres at the front
        # (ideal swivel casters), wheels on revolute axles at the back.
        wx, wy = hl - 0.08, hw - 0.06
        for name, x, y in (('fl', wx, wy), ('fr', wx, -wy), ('bl', -wx, wy), ('br', -wx, -wy)):
            box(s, f'{b}/caster_plate_{name}', (0.10, 0.08, 0.006),
                (x, y, top - thick - 0.003), metal, collide=False)
            box(s, f'{b}/caster_swivel_{name}', (0.05, 0.05, 0.02),
                (x, y, top - thick - 0.016), metal, collide=False)
            for k, dy in (('a', 0.022), ('b', -0.022)):
                box(s, f'{b}/caster_fork_{name}{k}', (0.03, 0.005, top - thick - r - 0.026),
                    (x, y + dy, (top - thick - 0.026 + r) / 2), metal, collide=False)
        for side, y in (('l', wy), ('r', -wy)):
            sph = UsdGeom.Sphere.Define(s, f'{b}/caster_{side}')
            sph.CreateRadiusAttr(r)
            sph.CreatePurposeAttr('guide')
            UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3d(wx, y, r))
            UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
            self.bind_material(sph.GetPrim(), self.caster_mat)
            tube_y(s, f'{b}/caster_wheel_{side}', r, 0.032, (wx, y, r), tyre)
        for side, y in (('l', wy), ('r', -wy)):
            w = s.DefinePrim(f'{CART}/wheel_{side}', 'Xform')
            place(w, (-wx, y, r))
            UsdPhysics.RigidBodyAPI.Apply(w)
            UsdPhysics.MassAPI.Apply(w).CreateMassAttr(0.5)
            sph = UsdGeom.Sphere.Define(s, f'{CART}/wheel_{side}/collider')
            sph.CreateRadiusAttr(r)
            sph.CreatePurposeAttr('guide')
            UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
            self.bind_material(sph.GetPrim(), self.tyre_mat)
            tube_y(s, f'{CART}/wheel_{side}/tyre', r, 0.032, (0.0, 0.0, 0.0), tyre)
            j = UsdPhysics.RevoluteJoint.Define(s, f'{CART}/axle_{side}')
            j.CreateBody0Rel().SetTargets([b])
            j.CreateBody1Rel().SetTargets([f'{CART}/wheel_{side}'])
            j.CreateAxisAttr('Y')
            j.CreateLocalPos0Attr(Gf.Vec3f(-wx, y, r))
            j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        # Hook-bar coupling between the robot's and the cart's articulations. Joints
        # added while simulating are not picked up, so it exists from the start and is
        # only enabled (with its frames set to the current poses) when the hook catches.
        latch = UsdPhysics.RevoluteJoint.Define(s, LATCH_JOINT)
        latch.CreateBody0Rel().SetTargets(['/World/amr/hitch_pitch_link'])
        latch.CreateBody1Rel().SetTargets([b])
        latch.CreateAxisAttr('Y')   # the bar turns freely in the hook
        latch.CreateLocalPos0Attr(Gf.Vec3f(*TIP_IN_PITCH))
        latch.CreateLocalPos1Attr(Gf.Vec3f(*self.cfg['cart']['tow_bar_in_cart']))
        latch.CreateExcludeFromArticulationAttr(True)
        latch.CreateJointEnabledAttr(False)
        latch.CreateCollisionEnabledAttr(False)

    def _build_charger(self):
        """Build a WiBotic TC-200 antenna and TR-302 transmitter stand (datasheet sizes)."""
        s = self.stage
        root = s.DefinePrim(CHARGER, 'Xform')
        place(root, (self.charger_pose[0], self.charger_pose[1], 0.0), self.charger_pose[2])
        black, metal, white = (0.05, 0.05, 0.05), (0.55, 0.57, 0.6), (0.92, 0.92, 0.92)
        tz = TX_IN_CHARGER[2]
        box(s, f'{CHARGER}/tc200_antenna', (0.022, 0.215, 0.215), (-0.011, 0.0, tz), black)
        box(s, f'{CHARGER}/stand_panel', (0.008, 0.50, 0.32), (-0.026, 0.0, 0.16), metal)
        box(s, f'{CHARGER}/stand_foot', (0.30, 0.50, 0.01), (-0.15, 0.0, 0.005), metal)
        box(s, f'{CHARGER}/tr302', (0.095, 0.32, 0.147), (-0.08, 0.0, 0.0835), metal)
        t = self.cfg['charger']
        for marker_id, y in t['markers'].items():   # white backing boards for the markers
            box(s, f'{CHARGER}/board_{marker_id}', (0.022, 0.13, 0.13),
                (-0.011, float(y), t['marker_height']), white)
        self._markers(CHARGER, 'charger', 0.001)

    # ---- runtime --------------------------------------------------------------
    def attach_ros(self, node, articulation_cls, world):
        self.node = node
        self.pub_gps = node.create_publisher(PoseStamped, 'indoor_gps/cart', 10)
        self.pub_battery = node.create_publisher(BatteryState, 'battery_state', 10)
        self.pub_latched = node.create_publisher(Bool, 'cart/latched', 10)
        node.create_service(SetBool, 'cart/latch', self._on_latch)
        if self.cart_pose is not None:
            self.cart = world.scene.add(articulation_cls(CART, name='cart'))

    def hook_tip(self, robot_pose, q_yaw, q_pitch):
        """Return the hook tip (hitch_coupling_link) pose from the robot pose and hitch angles."""
        pos, quat = robot_pose
        R = quat_wxyz_to_matrix(quat)
        Ry = R @ rot_z(q_yaw)
        Rp = Ry @ rot_y(q_pitch)
        p = np.asarray(pos) + R @ YAW_IN_BASE + Ry @ PITCH_IN_YAW + Rp @ TIP_IN_PITCH
        return p, Rp

    def tow_bar(self):
        pos, quat = self.cart.get_world_pose()
        R = quat_wxyz_to_matrix(quat)
        return np.asarray(pos) + R @ np.array(self.cfg['cart']['tow_bar_in_cart']), R, pos, quat

    def _on_latch(self, req, res):
        if self.cart is None:
            res.success, res.message = False, 'no cart in this world'
            return res
        if not req.data:
            self.latch_request = False
            res.success, res.message = True, 'released'
            return res
        dist = self._hook_offset()
        if dist > LATCH_TOLERANCE:
            res.success = False
            res.message = f'hook is {dist * 100:.1f} cm from the tow bar (max 3 cm)'
            return res
        self.latch_request = True
        res.success, res.message = True, f'latched ({dist * 100:.1f} cm)'
        return res

    def set_robot_state_fn(self, fn, pitch_target_fn):
        self._robot_state = fn
        self._pitch_target = pitch_target_fn

    def _auto_latch(self):
        pitch = self._pitch_target()
        if self.latched and pitch < RELEASE_PITCH:
            self.latch_request = False
        elif not self.latched and pitch > ENGAGE_PITCH:
            if self._hook_offset() <= LATCH_TOLERANCE:
                self.latch_request = True

    def _hook_offset(self):
        """Horizontal offset between the hook mouth (level hook) and the tow bar [m]."""
        pose, q_yaw, _ = self._robot_state()
        tip, _ = self.hook_tip(pose, q_yaw, 0.0)
        return float(np.linalg.norm((tip - self.tow_bar()[0])[:2]))

    def apply_requests(self):
        """Create/remove the latch joint. Called from the main loop, not from a callback."""
        if self.latch_request is None:
            return
        want, self.latch_request = self.latch_request, None
        if want and not self.latched:
            (pos, quat), qy, qp = self._robot_state()
            tip, Rp = self.hook_tip((pos, quat), qy, qp)
            _, Rc, cpos, _ = self.tow_bar()
            j = UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath(LATCH_JOINT))
            j.GetLocalPos1Attr().Set(Gf.Vec3f(*(Rc.T @ (tip - np.asarray(cpos)))))
            rel = Gf.Matrix3d(*(Rc.T @ Rp).T.flatten())  # pitch-link axes in the cart frame
            q = Gf.Matrix4d(rel, Gf.Vec3d(0)).ExtractRotationQuat()
            j.CreateLocalRot1Attr().Set(Gf.Quatf(q))
            j.GetJointEnabledAttr().Set(True)
            self.latched = True
        elif not want and self.latched:
            UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath(LATCH_JOINT)) \
                .GetJointEnabledAttr().Set(False)
            self.latched = False

    def publish(self, t, dt, robot_pose):
        """Indoor GPS, battery and latch state; call at a few Hz."""
        sec, nsec = int(t), int((t - int(t)) * 1e9)
        if self.cart is not None:
            if self.latch_request is None:
                self._auto_latch()
            pos, quat = self.cart.get_world_pose()
            w, x, y, z = quat
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            yaw += self.rng.normal(0.0, math.radians(1.0))
            msg = PoseStamped()
            msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nsec
            msg.header.frame_id = 'map'
            msg.pose.position.x = float(pos[0] + self.rng.normal(0.0, 0.02))
            msg.pose.position.y = float(pos[1] + self.rng.normal(0.0, 0.02))
            msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.pub_gps.publish(msg)
            self.pub_latched.publish(Bool(data=self.latched))
        self.charging = self.charger_pose is not None and self._rx_on_pad(robot_pose)
        current = 5.0 if self.charging else -2.0       # A; 20 Ah pack
        self.battery = min(1.0, max(0.0, self.battery + current * dt / (20.0 * 3600.0)))
        bat = BatteryState()
        bat.header.stamp.sec, bat.header.stamp.nanosec = sec, nsec
        bat.voltage = 22.0 + 4.0 * self.battery
        bat.current = current
        bat.percentage = self.battery
        bat.capacity = bat.design_capacity = 20.0
        bat.power_supply_status = (BatteryState.POWER_SUPPLY_STATUS_CHARGING if self.charging
                                   else BatteryState.POWER_SUPPLY_STATUS_DISCHARGING)
        bat.present = True
        self.pub_battery.publish(bat)

    def _rx_on_pad(self, robot_pose):
        pos, quat = robot_pose
        rx = np.asarray(pos) + quat_wxyz_to_matrix(quat) @ RX_IN_BASE
        cx, cy, cyaw = self.charger_pose
        d = rot_z(cyaw).T @ (rx - np.array([cx, cy, 0.0])) - TX_IN_CHARGER
        w, x, y, z = quat
        heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        misalign = abs(math.atan2(math.sin(heading - cyaw - math.pi),
                                  math.cos(heading - cyaw - math.pi)))
        return -0.005 <= d[0] <= 0.04 and abs(d[1]) <= 0.04 and misalign < math.radians(10)
