"""Isaac Sim 5.1 simulation of the towing AMR with the real robot's ROS 2 interface.

Run through run_sim.sh (it sets up the ROS 2 environment and expands the xacro):
  ./run_sim.sh [--headless] [--env warehouse|empty] [--no-cameras]

Then on the ROS side:
  ros2 launch amr_bringup sim.launch.py
"""

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser()
ap.add_argument('--urdf', required=True, help='expanded URDF with absolute mesh paths')
ap.add_argument('--usd-dir', required=True, help='cache directory for the converted robot')
ap.add_argument('--headless', action='store_true')
ap.add_argument('--env', default='warehouse', choices=['warehouse', 'empty'])
ap.add_argument('--env-usd', nargs='+', metavar='USD',
                help='custom environment instead of --env: USD/USDZ files referenced into '
                     'the stage, e.g. a 3DGRUT NuRec .usdz (docs/gaussian_splatting.md)')
ap.add_argument('--env-collision', metavar='USD',
                help='mesh of the custom environment (from the lidar map): static colliders, '
                     'hidden from cameras unless --env-collision-visible')
ap.add_argument('--env-collision-visible', action='store_true')
ap.add_argument('--no-cameras', action='store_true')
ap.add_argument('--no-lidar', action='store_true')
ap.add_argument('--chase-camera', action='store_true',
                help='publish a third-person view on /sim/chase_camera/image_raw')
ap.add_argument('--overview-camera', type=float, nargs=6,
                metavar=('X', 'Y', 'Z', 'TX', 'TY', 'TZ'),
                help='fixed camera from (X,Y,Z) towards (TX,TY,TZ) on '
                     '/sim/overview_camera/image_raw')
ap.add_argument('--no-realtime', action='store_true', help='run as fast as possible')
ap.add_argument('--selftest', action='store_true',
                help='drive a scripted sequence without ROS input, print results and exit')
ap.add_argument('--targets', help='amr_docking docking_targets.yaml (enables cart/charger)')
ap.add_argument('--markers', help='directory with the aruco_<id>.png markers')
ap.add_argument('--cart-pose', type=float, nargs=3, default=[3.0, -7.5, math.pi / 2],
                metavar=('X', 'Y', 'YAW'), help='cart frame in the world/map frame')
ap.add_argument('--charger-pose', type=float, nargs=3, default=[8.3, -1.0, math.pi],
                metavar=('X', 'Y', 'YAW'), help='charger target frame in the world/map frame')
ap.add_argument('--corridor', type=float, nargs='+', metavar='W',
                help='narrow-passage experiment: corridor width [m], optionally followed '
                     'by X0 X1 Y (default 3.0 5.5 -4.5), as two walls 0.6 m high')
ap.add_argument('--cart-mass', type=float, default=22.0,
                help='cart mass incl. load [kg] (experiments vary it)')
ap.add_argument('--cart-wheels', choices=('trailer', 'caster4'), default='trailer',
                help='cart wheels: trailer = swivel casters at the front and fixed '
                     'axles at the back (a heading the hitch angle is measured against), '
                     'caster4 = swivel casters all round (a free body that is pushed)')
ap.add_argument('--no-cart', action='store_true')
ap.add_argument('--no-charger', action='store_true')
ap.add_argument('--reimport', action='store_true', help='reconvert the URDF even if cached')
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.syntheticdata as osd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

enable_extension('isaacsim.ros2.bridge')
app.update()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mid360 import create_mid360  # noqa: E402, I100
from ros_interface import SimRosInterface  # noqa: E402
from world import box, DockingWorld  # noqa: E402

ROBOT = '/World/amr'
# Same values as amr_base/config/base.yaml and amr_hitch/config/hitch.yaml.
CFG = {
    'wheel_radius': 0.100, 'wheel_separation': 0.3742,
    'max_linear': 0.5, 'max_angular': 1.5, 'cmd_vel_timeout': 0.5,
    'hitch_limits': (-1.5708, 1.5708),
    'tyre_contact_width': 0.012,  # m, see setup_robot
    'wheel_max_torque': 11.0,   # Nm, MDH300 rated torque (peak not published)
    'wheel_damping': 1000.0,      # Nm s/rad: stiff velocity loop like the motor driver's
    'hitch_kp': 50.0, 'hitch_kd': 2.0,  # Nm/rad, Nm s/rad
    'hitch_max_torque': 10.6,           # XM540-W270 stall torque
}
WHEELS = ['left_wheel_joint', 'right_wheel_joint']
HITCH = ['hitch_yaw_joint', 'hitch_pitch_joint']
# Render rate = lidar scan rate, so each point cloud is one full revolution.
PHYSICS_HZ, RENDER_HZ, STATE_HZ = 200.0, 10.0, 50.0


def yaw_quat(yaw):
    """Quaternion (w, x, y, z) for a rotation about z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def import_robot():
    usd = os.path.join(args.usd_dir, 'amr.usd')
    if os.path.exists(usd) and not args.reimport and \
            os.path.getmtime(usd) > os.path.getmtime(args.urdf):
        return usd
    _, cfg = omni.kit.commands.execute('URDFCreateImportConfig')
    cfg.merge_fixed_joints = False   # keep sensor frames as prims
    cfg.fix_base = False
    cfg.make_default_prim = True
    cfg.create_physics_scene = False
    cfg.import_inertia_tensor = True
    cfg.self_collision = False
    cfg.collision_from_visuals = False
    cfg.distance_scale = 1.0
    omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=args.urdf,
                              import_config=cfg, dest_path=usd)
    return usd


def physics_material(stage, path, static, dynamic, combine):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr(combine)
    return mat


def deinstance(prim):
    """Make prim and everything below it editable.

    The importer instances visual/collision geometry, and instance proxies cannot be
    edited, so clear 'instanceable' on the enclosing instance and on nested ones.
    """
    path, stage = prim.GetPath(), prim.GetStage()
    p = prim
    while p and p.GetPath() != Sdf.Path.absoluteRootPath:
        if p.IsInstance():
            p.SetInstanceable(False)
        p = p.GetParent()
    prim = stage.GetPrimAtPath(path)
    while True:
        nested = [q for q in Usd.PrimRange(prim) if q.IsInstance()]
        if not nested:
            return prim
        for q in nested:
            q.SetInstanceable(False)


def colliders(prim):
    found = [q for q in Usd.PrimRange(deinstance(prim)) if q.HasAPI(UsdPhysics.CollisionAPI)]
    assert found and not any(q.IsInstanceProxy() for q in found), prim.GetPath()
    return found


def bind_physics_material(prim, mat):
    for q in colliders(prim):
        UsdShade.MaterialBindingAPI.Apply(q).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, 'physics')


def setup_robot(stage):
    # Links without <inertial> in the URDF (frames, caster spheres) would otherwise get
    # PhysX default masses; their mass is already part of base_link.
    urdf = ET.parse(args.urdf).getroot()
    massless = [link.get('name') for link in urdf.findall('link') if link.find('inertial') is None]
    for name in massless:
        mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(f'{ROBOT}/{name}'))
        mass.CreateMassAttr(1e-3)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))
    # The importer creates 'acceleration' drives, whose gains scale with link inertia
    # (far too weak for the light hitch links). Use torque drives; gains are set later.
    for joint in WHEELS + HITCH:
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f'{ROBOT}/joints/{joint}'), 'angular')
        drive.GetTypeAttr().Set('force')
    tyre = physics_material(stage, '/World/Physics/tyre', 1.0, 0.9, 'max')
    caster = physics_material(stage, '/World/Physics/caster', 0.0, 0.0, 'min')
    for side in ('left', 'right'):
        col = stage.GetPrimAtPath(f'{ROBOT}/{side}_wheel_link/collisions')
        bind_physics_material(col, tyre)
        # True cylinder instead of a convex hull: smooth rolling, no polygon chatter.
        # The tyre has a round profile (CAD torus, tube radius 23 mm), so it touches the
        # floor in a narrow band, not across its 46 mm width; a full-width contact
        # would add scrub torque when turning in place.
        for q in colliders(col):
            if q.IsA(UsdGeom.Cylinder):
                q.CreateAttribute('physxCollisionCustomGeometry',
                                  Sdf.ValueTypeNames.Bool).Set(True)
                UsdGeom.Cylinder(q).GetHeightAttr().Set(CFG['tyre_contact_width'])
    for name in ('front_caster_link', 'rear_caster_link'):
        bind_physics_material(stage.GetPrimAtPath(f'{ROBOT}/{name}/collisions'), caster)
    linearize_cad_colours(stage)


def linearize_cad_colours(stage):
    """Show the CAD colours as designed.

    The meshes carry the CATIA colours as sRGB values (what RViz displays directly),
    but Omniverse treats material colours as linear and brightens them on output, so a
    dark grey part would render light grey. Convert them to linear once.
    """
    def to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    for link in stage.GetPrimAtPath(ROBOT).GetChildren():
        visuals = stage.GetPrimAtPath(link.GetPath().AppendChild('visuals'))
        if not visuals:
            continue
        for q in Usd.PrimRange(deinstance(visuals)):
            if q.GetTypeName() != 'Shader':
                continue
            attr = q.GetAttribute('inputs:diffuse_color_constant')
            if attr and attr.Get() is not None and not q.GetCustomDataByKey('amr:linearized'):
                attr.Set(Gf.Vec3f(*(to_linear(c) for c in attr.Get())))
                q.SetCustomDataByKey('amr:linearized', True)


def add_custom_environment(stage):
    """Real-site twin: visuals (e.g. Gaussian splats) + a lidar-map mesh for physics."""
    for i, path in enumerate(args.env_usd or []):
        add_reference_to_stage(os.path.abspath(os.path.expanduser(path)), f'/World/env_{i}')
    # Ground plane for physics at z = 0 (the capture is aligned so the floor is z = 0),
    # hidden so the reconstructed floor shows.
    from isaacsim.core.api.objects import GroundPlane
    GroundPlane('/World/ground', size=200.0, visible=False)
    if args.env_collision:
        col = '/World/env_collision'
        add_reference_to_stage(os.path.abspath(os.path.expanduser(args.env_collision)), col)
        for q in Usd.PrimRange(stage.GetPrimAtPath(col)):
            if q.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(q)
                UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr('none')
                if not args.env_collision_visible:
                    # Hidden from cameras but still hit by the RTX lidar (which ignores
                    # invisible prims).
                    UsdGeom.PrimvarsAPI(q).CreatePrimvar(
                        'hideForCamera', Sdf.ValueTypeNames.Bool).Set(True)
    light = stage.DefinePrim('/World/light', 'DomeLight')
    light.CreateAttribute('inputs:intensity', Sdf.ValueTypeNames.Float).Set(1000.0)


def add_corridor(stage, params):
    """Build a passage that is the only way across (narrow-passage experiments).

    Two parallel walls form the passage itself (x0..x1 at y +- width/2), and wings at
    the x0 face run out to the edges of the driveable area. Without the wings the walls
    are two free-standing obstacles in open floor and Nav2 simply plans around them, so
    nothing is ever measured inside the passage.

    params: WIDTH [X0 X1 Y].
    """
    defaults = [3.0, 5.5, -4.5]
    width = params[0]
    x0, x1, y = (list(params[1:4]) + defaults[len(params) - 1:])[:3]
    length, height, thickness = x1 - x0, 0.6, 0.1
    y_min, y_max = -9.0, 1.0          # beyond the driveable part of the warehouse
    grey = (0.35, 0.35, 0.37)
    stage.DefinePrim('/World/corridor', 'Xform')
    half = (width + thickness) / 2.0
    for side, name in ((1, 'left'), (-1, 'right')):
        box(stage, f'/World/corridor/wall_{name}', (length, thickness, height),
            (0.5 * (x0 + x1), y + side * half, height / 2.0), grey)
    for side, name, far in ((1, 'wing_left', y_max), (-1, 'wing_right', y_min)):
        span = abs(far - (y + side * half))
        box(stage, f'/World/corridor/{name}', (thickness, span, height),
            (x0, 0.5 * (far + y + side * half), height / 2.0), grey)
    return (width, x0, x1, y)


def add_environment(stage):
    if args.env_usd or args.env_collision:
        add_custom_environment(stage)
        return
    if args.env == 'warehouse':
        # get_assets_root_path() raises when the asset server cannot be reached rather
        # than returning None, and a transient network hiccup then kills the whole run.
        try:
            root = get_assets_root_path()
        except RuntimeError as e:
            print(f'[amr_sim] asset server: {e}', flush=True)
            root = None
        if root:
            add_reference_to_stage(root + '/Isaac/Environments/Simple_Warehouse/warehouse.usd',
                                   '/World/env')
            return
        print('[amr_sim] asset server unreachable, using an empty ground plane', flush=True)
    from isaacsim.core.api.objects import GroundPlane
    GroundPlane('/World/ground', size=50.0)
    light = stage.DefinePrim('/World/light', 'DistantLight')
    light.CreateAttribute('inputs:intensity', Sdf.ValueTypeNames.Float).Set(3000.0)


def add_camera(stage, link, name, topic_ns, writers):
    """D435-like camera: colour + aligned depth at the realsense camera_link origin."""
    path = f'{ROBOT}/{link}/{name}'
    cam = UsdGeom.Camera.Define(stage, path)
    # USD cameras look along -Z with +Y up; the link has x forward, z up.
    rot = Gf.Matrix3d(0, -1, 0, 0, 0, 1, -1, 0, 0)  # rows: cam X, Y, Z in link coords
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Matrix4d(rot, Gf.Vec3d(0)).ExtractRotationQuat()))
    w, h, hfov = 640, 480, 69.0   # D435 colour FOV 69 x 42 deg
    aperture = 2.0 * math.tan(math.radians(hfov) / 2.0) * 1.93  # focal length 1.93 mm
    cam.CreateFocalLengthAttr(1.93)
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 20.0))
    rp = rep.create.render_product(path, (w, h))
    frame = f'{topic_ns}_color_optical_frame'
    rgb = osd.SyntheticData.convert_sensor_type_to_rendervar('Rgb')
    depth = osd.SyntheticData.convert_sensor_type_to_rendervar('DistanceToImagePlane')
    for writer_name, topic in ((f'{rgb}ROS2PublishImage', f'{topic_ns}/color/image_raw'),
                               (f'{depth}ROS2PublishImage', f'{topic_ns}/depth/image_rect_raw')):
        wr = rep.writers.get(writer_name)
        wr.initialize(topicName=topic, frameId=frame)
        wr.attach([rp])
        writers.append(wr)
    try:
        from isaacsim.ros2.bridge.impl.camera_info_utils import read_camera_info
        info, _ = read_camera_info(render_product_path=rp.path)
        wr = rep.writers.get('ROS2PublishCameraInfo')
        wr.initialize(topicName=f'{topic_ns}/color/camera_info', frameId=frame,
                      width=info.width, height=info.height, projectionType=info.distortion_model,
                      k=info.k.reshape([1, 9]), r=info.r.reshape([1, 9]),
                      p=info.p.reshape([1, 12]), physicalDistortionModel=info.distortion_model,
                      physicalDistortionCoefficients=info.d)
        wr.attach([rp])
        writers.append(wr)
    except Exception as exc:  # camera info is optional
        print(f'[amr_sim] camera_info for {topic_ns} disabled: {exc}', flush=True)
    # optical frame: z forward, x right, y down
    return (link, frame, (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5))


def add_view_camera(stage, writers, path, eye, target, topic, resolution=(960, 540)):
    """Demo camera looking from eye to target (in its parent's frame), RGB on `topic`."""
    cam = UsdGeom.Camera.Define(stage, path)
    eye, target = Gf.Vec3d(*eye), Gf.Vec3d(*target)
    z = (eye - target).GetNormalized()            # USD cameras look along -Z
    x = Gf.Cross(Gf.Vec3d(0, 0, 1), z).GetNormalized()
    y = Gf.Cross(z, x)
    rot = Gf.Matrix3d(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(eye)
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Matrix4d(rot, Gf.Vec3d(0)).ExtractRotationQuat()))
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(24.0)
    cam.CreateVerticalApertureAttr(24.0 * resolution[1] / resolution[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    rp = rep.create.render_product(path, resolution)
    rgb = osd.SyntheticData.convert_sensor_type_to_rendervar('Rgb')
    wr = rep.writers.get(f'{rgb}ROS2PublishImage')
    wr.initialize(topicName=topic, frameId=path.split('/')[-1])
    wr.attach([rp])
    writers.append(wr)


def add_lidar(stage, writers):
    path = f'{ROBOT}/livox_frame/mid360'
    create_mid360(stage, path)
    rp = rep.create.render_product(path, (1, 1), name='mid360')
    # One render frame = one 10 Hz revolution, like the real driver's 100 ms frames.
    # (The ...PointCloudBuffer writer accumulates wrongly in 5.1.)
    wr = rep.writers.get('RtxLidarROS2PublishPointCloud')
    wr.initialize(topicName='livox/lidar', frameId='livox_frame')
    wr.attach([rp])
    writers.append(wr)


def selftest(world, robot, ros, names, ctrl):
    """Scripted checks of the drive and hitch without ROS clients; exit code 1 on failure."""
    def run(sec, v=0.0, w=0.0):
        t_end = world.current_time + sec
        while world.current_time < t_end:
            ros.cmd, ros.cmd_time = (v, w), world.current_time
            world.step(render=False)

    def pose():
        pos, (qw, qx, qy, qz) = robot.get_world_pose()
        return pos[0], pos[1], math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    results = []

    def check(name, value, expected, tol):
        ok = abs(value - expected) <= tol
        results.append(ok)
        print(f'[selftest] {"PASS" if ok else "FAIL"} {name}: {value:.3f} '
              f'(expected {expected} +- {tol})', flush=True)

    run(1.0)
    run(2.0, 0.3, 0.0)
    p1 = pose()
    run(2.0, 0.3, 0.0)
    p2 = pose()
    speed = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 2.0
    check('linear speed at 0.3 m/s [m/s]', speed, 0.3, 0.02)
    run(1.0)
    run(1.0, 0.0, 0.5)
    y1 = pose()[2]
    run(2.0, 0.0, 0.5)
    y2 = pose()[2]
    check('yaw rate at 0.5 rad/s [rad/s]', (y2 - y1) / 2.0, 0.5, 0.05)
    run(1.0, 0.3, 0.0)
    ros.cmd_time = -1e9
    p3 = pose()
    run(1.0)
    p4 = pose()
    check('cmd_vel timeout stop: drift [m]', math.hypot(p4[0] - p3[0], p4[1] - p3[1]), 0.0, 0.05)
    ros.hitch_target.update({'hitch_yaw_joint': 0.6, 'hitch_pitch_joint': 0.3})
    run(2.0)
    q = robot.get_joint_positions()
    check('hitch yaw [rad]', q[names.index('hitch_yaw_joint')], 0.6, 0.02)
    check('hitch pitch [rad]', q[names.index('hitch_pitch_joint')], 0.3, 0.02)
    print(f'[selftest] {sum(results)}/{len(results)} passed', flush=True)
    return all(results)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ,
                  rendering_dt=1.0 / RENDER_HZ)
    stage = world.stage
    add_environment(stage)
    if args.corridor:
        geometry = add_corridor(stage, list(args.corridor))
        print(f'[amr_sim] corridor width {geometry[0]:.2f} m at y {geometry[3]:.2f}',
              flush=True)
    add_reference_to_stage(import_robot(), ROBOT)
    UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT)).ClearXformOpOrder()
    UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT)).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.002))
    setup_robot(stage)

    writers, optical = [], []
    if not args.no_lidar:
        add_lidar(stage, writers)
    if not args.no_cameras:
        optical.append(add_camera(stage, 'front_camera_link', 'd435', 'front_camera', writers))
        optical.append(add_camera(stage, 'rear_camera_link', 'd435', 'rear_camera', writers))

    if args.chase_camera:   # follows the robot
        add_view_camera(stage, writers, f'{ROBOT}/base_footprint/chase_camera',
                        (-1.1, -0.9, 0.8), (0.15, 0.0, 0.2), 'sim/chase_camera/image_raw',
                        (1280, 720))
    if args.overview_camera:  # fixed in the world
        e = args.overview_camera
        add_view_camera(stage, writers, '/World/overview_camera', e[:3], e[3:],
                        'sim/overview_camera/image_raw')

    docking = None
    if args.targets:
        tyre = physics_material(stage, '/World/Physics/cart_tyre', 0.9, 0.8, 'average')
        caster = stage.GetPrimAtPath('/World/Physics/caster')
        docking = DockingWorld(stage, args.targets, args.markers,
                               None if args.no_cart else args.cart_pose,
                               None if args.no_charger else args.charger_pose,
                               (tyre, UsdShade.Material(caster)),
                               lambda prim, mat: UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                                   mat, UsdShade.Tokens.weakerThanDescendants, 'physics'),
                               cart_mass=args.cart_mass,
                               cart_wheels=args.cart_wheels)

    robot = world.scene.add(SingleArticulation(ROBOT, name='amr'))
    ros = SimRosInterface(CFG, HITCH)
    if docking:
        docking.attach_ros(ros.node, SingleArticulation, world)
    world.reset()
    names = robot.dof_names
    wheel_idx = np.array([names.index(j) for j in WHEELS])
    hitch_idx = np.array([names.index(j) for j in HITCH])
    ctrl = robot.get_articulation_controller()
    kps, kds = np.zeros(len(names)), np.zeros(len(names))
    kds[wheel_idx] = CFG['wheel_damping']
    kps[hitch_idx], kds[hitch_idx] = CFG['hitch_kp'], CFG['hitch_kd']
    ctrl.set_gains(kps=kps, kds=kds)
    efforts = np.full(len(names), 1e3)
    efforts[wheel_idx] = CFG['wheel_max_torque']
    efforts[hitch_idx] = CFG['hitch_max_torque']
    ctrl.set_max_efforts(efforts)

    ros.publish_static([(p, c, xyz, q) for p, c, xyz, q in optical])
    hitch_state = (True, False)   # (torque on, yaw passive)
    state_every = int(PHYSICS_HZ / STATE_HZ)
    step = {'n': 0}
    world_every = int(STATE_HZ / 5)   # GPS, battery: 5 Hz (GPS jitter like 8 Hz beacons)

    def robot_state():
        q = robot.get_joint_positions()
        return (robot.get_world_pose(), float(q[names.index('hitch_yaw_joint')]),
                float(q[names.index('hitch_pitch_joint')]))

    if docking:
        docking.set_robot_state_fn(robot_state, lambda: ros.hitch_target['hitch_pitch_joint'])

    def apply_set_state(request):
        """Place the robot (and cart) for the next trial of a repeated experiment.

        Only for experiments (/sim/set_state); the real robot has no such interface.
        A latched cart is released first: the latch joint frames are set on engage.
        """
        (rx, ry, ryaw), cart = request
        if docking and docking.latched:
            docking.latch_request = False
            docking.apply_requests()
        ros.cmd, ros.cmd_time = (0.0, 0.0), -1e9
        for j in HITCH:
            ros.hitch_target[j] = 0.0
        zeros3 = np.zeros(3)
        robot.set_world_pose(np.array([rx, ry, 0.002]), yaw_quat(ryaw))
        robot.set_linear_velocity(zeros3)
        robot.set_angular_velocity(zeros3)
        robot.set_joint_positions(np.zeros(len(names)))
        robot.set_joint_velocities(np.zeros(len(names)))
        if cart is not None and docking and docking.cart is not None:
            cx, cy, cyaw = cart
            docking.cart.set_world_pose(np.array([cx, cy, 0.0]), yaw_quat(cyaw))
            docking.cart.set_linear_velocity(zeros3)
            docking.cart.set_angular_velocity(zeros3)

    def on_physics(dt):
        t = world.current_time
        ros.set_time(t)
        step['n'] += 1
        wl, wr = ros.wheel_targets()
        robot.apply_action(ArticulationAction(joint_velocities=np.array([wl, wr]),
                                              joint_indices=wheel_idx))
        robot.apply_action(ArticulationAction(
            joint_positions=np.array([ros.hitch_target[j] for j in HITCH]),
            joint_indices=hitch_idx))
        if step['n'] % state_every:
            return
        ros.spin_once()  # commands are handled at STATE_HZ, independent of rendering
        pos, quat = robot.get_world_pose()  # articulation root = base_footprint
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        lin, ang = robot.get_linear_velocity(), robot.get_angular_velocity()
        vx = math.cos(yaw) * lin[0] + math.sin(yaw) * lin[1]
        q, qd = robot.get_joint_positions(), robot.get_joint_velocities()
        tau = np.array(robot.get_measured_joint_efforts(), dtype=float)
        # Hitch effort = servo drive torque (what the Dynamixel current reflects). The
        # measured joint effort misses loads from the cart latch (a non-articulation joint).
        for j, i in zip(HITCH, hitch_idx):
            tau[i] = np.clip(kps[i] * (ros.hitch_target[j] - q[i]) - kds[i] * qd[i],
                             -CFG['hitch_max_torque'], CFG['hitch_max_torque'])
        idx = list(wheel_idx) + list(hitch_idx)
        ros.publish_state(t, float(pos[0]), float(pos[1]), yaw, vx, float(ang[2]),
                          [names[i] for i in idx], [float(q[i]) for i in idx],
                          [float(qd[i]) for i in idx], [float(tau[i]) for i in idx])
        # /clock after the TF for that time, so nodes never ask for a future transform.
        ros.publish_clock(t)
        if docking and (step['n'] // state_every) % world_every == 0:
            docking.publish(t, world_every / STATE_HZ, (pos, quat))

    world.add_physics_callback('amr_control', on_physics)
    if args.selftest:
        ok = selftest(world, robot, ros, names, ctrl)
        os._exit(0 if ok else 1)  # app.close() can hang or crash after a headless run
    print('[amr_sim] running. Ctrl+C to stop.', flush=True)
    wall0, sim0 = time.time(), world.current_time
    try:
        while app.is_running():
            world.step(render=True)
            if docking:
                docking.apply_requests()
            if ros.set_state is not None:
                apply_set_state(ros.set_state)
                ros.set_state = None
            state = (ros.hitch_torque, ros.yaw_passive)
            if state != hitch_state:
                hitch_state = state
                q = robot.get_joint_positions()
                for j in HITCH:   # hold the current pose, like amr_hitch on torque-on
                    ros.hitch_target[j] = float(q[names.index(j)])
                for j in HITCH:
                    on = ros.hitch_torque and not (j == 'hitch_yaw_joint' and ros.yaw_passive)
                    kps[names.index(j)] = CFG['hitch_kp'] if on else 0.0
                    kds[names.index(j)] = CFG['hitch_kd'] if on else 0.05  # bearing friction
                ctrl.set_gains(kps=kps, kds=kds)
            if not args.no_realtime:
                ahead = (world.current_time - sim0) - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
    except KeyboardInterrupt:
        pass
    ros.shutdown()
    app.close()


try:
    main()
except Exception:  # Kit swallows the exit code otherwise
    import traceback
    traceback.print_exc()
    os._exit(1)
