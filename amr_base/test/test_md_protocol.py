import struct

from amr_base import md_protocol as md


def test_velocity_command_matches_cpp_layout():
    # Header {183, 184, 1}, PID 207, LEN 7, packed PID_PNT_VEL_CMD_Data, checksum.
    frame = md.velocity_command(-100, 200)
    assert frame == bytes([0xB7, 0xB8, 0x01, 0xCF, 0x07,
                           0x01, 0x9C, 0xFF, 0x01, 0xC8, 0x00, 0x02, 0x53])


def test_frames_sum_to_zero():
    for frame in (md.torque_off(), md.brake(), md.hall_type(10), md.encoder_ppr(1280),
                  md.request_main_data(), md.set_baudrate(2), md.velocity_command(5, -5)):
        assert sum(frame) & 0xFF == 0


def test_encoder_ppr_little_endian():
    assert md.encoder_ppr(1280)[5:7] == bytes([0x00, 0x05])


def test_velocity_command_saturates():
    frame = md.velocity_command(100000, -100000)
    assert struct.unpack_from('<h', frame, 6)[0] == 32767
    assert struct.unpack_from('<h', frame, 9)[0] == -32768


def _main_data_frame(rpm1=12, cur1=35, pos1=-7, rpm2=-3, cur2=10, pos2=2**31 - 1):
    payload = struct.pack('<hhBihhBi', rpm1, cur1, 0, pos1, rpm2, cur2, 1, pos2)
    head = bytes([md.PC_ID, md.MD_DRIVER_ID, 1, md.PID_PNT_MAIN_DATA, len(payload)]) + payload
    return head + bytes([md.checksum(head)])


def test_main_data_frame_length():
    assert len(_main_data_frame()) == md.MAIN_DATA_FRAME_LEN == 24


def test_parser_decodes_frame():
    data = md.FrameParser().feed(_main_data_frame())
    assert data.rpm_ch1 == 12 and data.rpm_ch2 == -3
    assert data.current_ch1 == 3.5 and data.current_ch2 == 1.0
    assert data.position_ch1 == -7 and data.position_ch2 == 2**31 - 1


def test_parser_resyncs_after_garbage_and_split_reads():
    parser = md.FrameParser()
    stream = b'\x01\xd2\x12\xff' + _main_data_frame(rpm1=1) + _main_data_frame(rpm1=2)
    assert parser.feed(stream[:10]) is None
    assert parser.feed(stream[10:]).rpm_ch1 == 2


def test_parser_rejects_bad_checksum():
    frame = bytearray(_main_data_frame())
    frame[-1] ^= 0xFF
    assert md.FrameParser().feed(bytes(frame)) is None
