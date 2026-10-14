"""ROS 2 driver for the 2-DOF (yaw, pitch) Dynamixel towing hitch."""

from amr_hitch import dxl_units as u
from dynamixel_sdk import (COMM_SUCCESS, GroupSyncRead, GroupSyncWrite, PacketHandler,
                           PortHandler)
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool


class HitchNode(Node):

    def __init__(self):
        super().__init__('amr_hitch')

        self.declare_parameter('port', '/dev/ttyDynamixel')
        self.declare_parameter('baudrate', 1000000)
        self.declare_parameter('joint_names', ['hitch_yaw_joint', 'hitch_pitch_joint'])
        self.declare_parameter('ids', [1, 2])
        self.declare_parameter('zero_ticks', [2048, 2048])
        self.declare_parameter('directions', [1, 1])
        self.declare_parameter('lower_limits', [-1.5708, -1.5708])
        self.declare_parameter('upper_limits', [1.5708, 1.5708])
        self.declare_parameter('operating_mode', 'position')
        self.declare_parameter('goal_current', 0)  # Raw units, current_based_position only.
        self.declare_parameter('profile_velocity', 0)  # Raw units, 0 = unlimited.
        self.declare_parameter('profile_acceleration', 0)
        self.declare_parameter('current_unit_mA', 2.69)  # XM430/XM540.
        self.declare_parameter('torque_on_start', True)
        self.declare_parameter('update_rate', 50.0)

        p = self.get_parameter
        self.names = list(p('joint_names').value)
        self.ids = list(p('ids').value)
        self.zero = list(p('zero_ticks').value)
        self.dir = list(p('directions').value)
        self.lower = list(p('lower_limits').value)
        self.upper = list(p('upper_limits').value)
        self.current_unit_a = p('current_unit_mA').value / 1000.0
        mode_name = p('operating_mode').value
        if mode_name not in u.OPERATING_MODES:
            raise ValueError(f'operating_mode must be one of {list(u.OPERATING_MODES)}')

        self.port = PortHandler(p('port').value)
        self.packet = PacketHandler(2.0)
        if not self.port.openPort() or not self.port.setBaudRate(p('baudrate').value):
            raise RuntimeError(f'Failed to open {p("port").value}.')

        for dxl_id in self.ids:
            self._write(1, dxl_id, u.ADDR_TORQUE_ENABLE, 0)
            self._write(1, dxl_id, u.ADDR_OPERATING_MODE, u.OPERATING_MODES[mode_name])
            self._write(4, dxl_id, u.ADDR_PROFILE_ACCELERATION, p('profile_acceleration').value)
            self._write(4, dxl_id, u.ADDR_PROFILE_VELOCITY, p('profile_velocity').value)
            if mode_name == 'current_based_position':
                self._write(2, dxl_id, u.ADDR_GOAL_CURRENT, p('goal_current').value)

        self.reader = GroupSyncRead(self.port, self.packet, u.ADDR_PRESENT_CURRENT,
                                    u.PRESENT_BLOCK_LEN)
        for dxl_id in self.ids:
            self.reader.addParam(dxl_id)
        self.writer = GroupSyncWrite(self.port, self.packet, u.ADDR_GOAL_POSITION, 4)

        self.torque_enabled = False
        if p('torque_on_start').value:
            self._set_torque(True)

        self.pub_joints = self.create_publisher(JointState, 'joint_states', 10)
        self.create_subscription(JointState, '~/command', self._on_command, 10)
        self.create_service(SetBool, '~/set_torque', self._on_set_torque)
        self.create_timer(1.0 / p('update_rate').value, self._update)
        self.get_logger().info(f'amr_hitch is up, mode={mode_name}, ids={self.ids}.')

    def _write(self, nbytes: int, dxl_id: int, addr: int, value: int) -> bool:
        fn = {1: self.packet.write1ByteTxRx, 2: self.packet.write2ByteTxRx,
              4: self.packet.write4ByteTxRx}[nbytes]
        result, error = fn(self.port, dxl_id, addr, value)
        if result != COMM_SUCCESS:
            self.get_logger().error(
                f'[ID {dxl_id}] addr {addr}: {self.packet.getTxRxResult(result)}')
            return False
        if error:
            self.get_logger().error(
                f'[ID {dxl_id}] addr {addr}: {self.packet.getRxPacketError(error)}')
            return False
        return True

    def _set_torque(self, enable: bool) -> bool:
        if enable:
            # Hold the current pose instead of jumping to a stale goal.
            self.reader.txRxPacket()
            for dxl_id in self.ids:
                if self.reader.isAvailable(dxl_id, u.ADDR_PRESENT_POSITION, 4):
                    present = self.reader.getData(dxl_id, u.ADDR_PRESENT_POSITION, 4)
                    self._write(4, dxl_id, u.ADDR_GOAL_POSITION, present)
        # Write every servo even if one fails, so a torque-off is never partial.
        results = [self._write(1, i, u.ADDR_TORQUE_ENABLE, int(enable)) for i in self.ids]
        ok = all(results)
        self.torque_enabled = enable and ok
        return ok

    def _on_set_torque(self, request, response):
        response.success = self._set_torque(request.data)
        response.message = 'torque ' + ('on' if self.torque_enabled else 'off')
        return response

    def _on_command(self, msg: JointState):
        if not self.torque_enabled:
            self.get_logger().warn('Ignoring command, torque is off.', throttle_duration_sec=2.0)
            return
        if len(msg.position) != len(msg.name):
            self.get_logger().warn('Command needs one position per joint name.')
            return
        self.writer.clearParam()
        for name, position in zip(msg.name, msg.position):
            if name not in self.names:
                self.get_logger().warn(f'Unknown joint "{name}".', throttle_duration_sec=2.0)
                continue
            i = self.names.index(name)
            ticks = u.rad_to_ticks(position, self.zero[i], self.dir[i],
                                   self.lower[i], self.upper[i])
            self.writer.addParam(self.ids[i], list((ticks & 0xFFFFFFFF).to_bytes(4, 'little')))
        result = self.writer.txPacket()
        if result != COMM_SUCCESS:
            self.get_logger().error(f'Sync write failed: {self.packet.getTxRxResult(result)}')

    def _update(self):
        result = self.reader.txRxPacket()
        if result != COMM_SUCCESS:
            self.get_logger().warn(f'Sync read failed: {self.packet.getTxRxResult(result)}',
                                   throttle_duration_sec=2.0)
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for i, dxl_id in enumerate(self.ids):
            if not self.reader.isAvailable(dxl_id, u.ADDR_PRESENT_CURRENT, u.PRESENT_BLOCK_LEN):
                continue
            pos = u.to_signed(self.reader.getData(dxl_id, u.ADDR_PRESENT_POSITION, 4), 4)
            vel = u.to_signed(self.reader.getData(dxl_id, u.ADDR_PRESENT_VELOCITY, 4), 4)
            cur = u.to_signed(self.reader.getData(dxl_id, u.ADDR_PRESENT_CURRENT, 2), 2)
            msg.name.append(self.names[i])
            msg.position.append(u.ticks_to_rad(pos, self.zero[i], self.dir[i]))
            msg.velocity.append(u.velocity_to_rad_s(vel, self.dir[i]))
            # Motor current in Amperes (not torque), signed like the joint.
            msg.effort.append(self.dir[i] * cur * self.current_unit_a)
        if msg.name:
            self.pub_joints.publish(msg)

    def shutdown(self):
        for dxl_id in self.ids:
            self.packet.write1ByteTxRx(self.port, dxl_id, u.ADDR_TORQUE_ENABLE, 0)
        self.port.closePort()


def main(args=None):
    rclpy.init(args=args)
    node = HitchNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
