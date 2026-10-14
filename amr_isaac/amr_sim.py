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
ap.add_argument('--no-cameras', action='store_true')
ap.add_argument('--no-lidar', action='store_true')
ap.add_argument('--chase-camera', action='store_true',
                help='publish a third-person view on /sim/chase_camera/image_raw')
ap.add_argument('--no-realtime', action='store_true', help='run as fast as possible')
ap.add_argument('--selftest', action='store_true',
                help='drive a scripted sequence without ROS input, print results and exit')
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


def add_environment(stage):
    if args.env == 'warehouse':
        root = get_assets_root_path()
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


def add_chase_camera(stage, writers, eye=(-1.1, -0.9, 0.8), target=(0.15, 0.0, 0.2)):
    """Third-person camera that follows the robot (for videos and demos)."""
    path = f'{ROBOT}/base_footprint/chase_camera'
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
    cam.CreateVerticalApertureAttr(13.5)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    rp = rep.create.render_product(path, (1280, 720))
    rgb = osd.SyntheticData.convert_sensor_type_to_rendervar('Rgb')
    wr = rep.writers.get(f'{rgb}ROS2PublishImage')
    wr.initialize(topicName='sim/chase_camera/image_raw', frameId='chase_camera')
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

    if args.chase_camera:
        add_chase_camera(stage, writers)

    robot = world.scene.add(SingleArticulation(ROBOT, name='amr'))
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

    ros = SimRosInterface(CFG, HITCH)
    ros.publish_static([(p, c, xyz, q) for p, c, xyz, q in optical])
    torque_on = True
    state_every = int(PHYSICS_HZ / STATE_HZ)
    step = {'n': 0}

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
        ros.publish_clock(t)
        ros.spin_once()  # commands are handled at STATE_HZ, independent of rendering
        pos, quat = robot.get_world_pose()  # articulation root = base_footprint
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        lin, ang = robot.get_linear_velocity(), robot.get_angular_velocity()
        vx = math.cos(yaw) * lin[0] + math.sin(yaw) * lin[1]
        q, qd = robot.get_joint_positions(), robot.get_joint_velocities()
        tau = robot.get_measured_joint_efforts()
        idx = list(wheel_idx) + list(hitch_idx)
        ros.publish_state(t, float(pos[0]), float(pos[1]), yaw, vx, float(ang[2]),
                          [names[i] for i in idx], [float(q[i]) for i in idx],
                          [float(qd[i]) for i in idx], [float(tau[i]) for i in idx])

    world.add_physics_callback('amr_control', on_physics)
    if args.selftest:
        ok = selftest(world, robot, ros, names, ctrl)
        os._exit(0 if ok else 1)  # app.close() can hang or crash after a headless run
    print('[amr_sim] running. Ctrl+C to stop.', flush=True)
    wall0, sim0 = time.time(), world.current_time
    try:
        while app.is_running():
            world.step(render=True)
            if ros.hitch_torque != torque_on:
                torque_on = ros.hitch_torque
                if torque_on:  # hold the current pose, like amr_hitch does on torque-on
                    q = robot.get_joint_positions()
                    ros.hitch_target.update({j: float(q[names.index(j)]) for j in HITCH})
                kps[hitch_idx] = CFG['hitch_kp'] if torque_on else 0.0
                kds[hitch_idx] = CFG['hitch_kd'] if torque_on else 0.05  # bearing friction
                ctrl.set_gains(kps=kps, kds=kds)
            if not args.no_realtime:
                ahead = (world.current_time - sim0) - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
    except KeyboardInterrupt:
        pass
    ros.shutdown()
    app.close()


main()
