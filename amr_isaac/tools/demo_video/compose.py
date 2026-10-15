"""Compose the recorded demo into a 2x2 captioned video: compose.py OUT_DIR [--speed 2.0].

Panels: overview, chase camera, RViz, docking camera (front until the cart phase, then
rear). Caption times come from OUT_DIR/events.txt written by record.sh.
"""
import argparse
import os
import subprocess

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--speed', type=float, default=2.0)
    ap.add_argument('--name', default='amr_docking_towing.mp4')
    a = ap.parse_args()
    os.chdir(a.out)
    ev, effort = {}, None
    for line in open('events.txt'):
        t, *what = line.split()
        if what[0] == 'effort':
            effort = float(what[1])
        else:
            ev.setdefault(what[0], float(t))
    rear_from = ev['cart_start']
    n = min(len(os.listdir('frames/front')), len(os.listdir('frames/rear')))
    os.makedirs('frames/dock', exist_ok=True)
    for i in range(n):
        link = f'frames/dock/f{i:05d}.jpg'
        if not os.path.lexists(link):
            src = 'front' if i / 10.0 < rear_from else 'rear'
            os.symlink(os.path.abspath(f'frames/{src}/f{i:05d}.jpg'), link)
    hook = f'hitch current {effort:.1f} Nm' if effort else 'hitch current check'
    caps = [
        (1.0, ev['charger_done'], '1  Auto-docking to the WiBotic charger   (Nav2 + ArUco)'),
        (ev['charger_done'], ev['undock_start'], '2  Wireless charging   (battery current +5 A)'),
        (ev['undock_start'], ev['cart_start'], '3  Undocking'),
        (ev['cart_start'], ev['cart_at_bar'],
         '4  Cart pose from indoor GPS  ->  navigate  ->  reverse-dock on ArUco'),
        (ev['cart_at_bar'], ev['tow_start'], f'5  Hook raised  -  {hook}  ->  cart coupled'),
        (ev['tow_start'], ev['tow_done'],
         '6  Towing the loaded cart   (footprint = robot + cart)'),
        (ev['tow_done'], ev['end'], '7  Release  -  hook lowered, robot drives away'),
    ]
    label = (f'drawtext=fontfile={FONT}:fontcolor=white:fontsize=22:box=1:'
             'boxcolor=black@0.55:boxborderw=6:x=10')
    cap = (f'drawtext=fontfile={FONT}:fontcolor=yellow:fontsize=30:box=1:boxcolor=black@0.75:'
           'boxborderw=12:x=(w-text_w)/2:y=h/2-text_h/2')
    captions = ','.join(f"{cap}:text='{txt}':enable='between(t,{t0:.2f},{t1:.2f})'"
                        for t0, t1, txt in caps)
    speed = f'{a.speed:g}'

    def panel(i, name, text, y='10', extra=''):
        txt = f"{label}:y={y}:text='{text}'"
        return f'[{i}:v]{extra}setpts=PTS-STARTPTS,{txt}[{name}];'
    dock = (f"{label}:y=h-text_h-10:text='Docking camera  -  front D435':"
            f"enable='lt(t,{rear_from})',"
            f"{label}:y=h-text_h-10:text='Docking camera  -  rear D435':"
            f"enable='gte(t,{rear_from})'")
    badge = (f'drawtext=fontfile={FONT}:fontcolor=white:fontsize=26:box=1:'
             f"boxcolor=red@0.75:boxborderw=8:x=w-text_w-14:y=14:text='x{speed}'")
    graph = (
        panel(0, 'a', 'Isaac Sim  -  warehouse overview', extra='scale=960:540,')
        + panel(1, 'b', 'Isaac Sim  -  chase camera', extra='scale=960:540,')
        + panel(2, 'c', 'RViz  -  SLAM map, Nav2 plan, footprint, cart GPS',
                y='h-text_h-34', extra='scale=960:540,')
        + f'[3:v]scale=720:540,pad=960:540:120:0:black,setpts=PTS-STARTPTS,{dock}[d];'
        + f'[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,{captions},'
        + f'{badge},setpts=PTS/{speed},fps=30[v]')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y',
                    '-framerate', '10', '-i', 'frames/overview/f%05d.jpg',
                    '-framerate', '10', '-i', 'frames/chase/f%05d.jpg', '-i', 'rviz.mp4',
                    '-framerate', '10', '-i', 'frames/dock/f%05d.jpg',
                    '-filter_complex', graph, '-map', '[v]', '-t', f'{ev["end"] / a.speed:.1f}',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                    '-movflags', '+faststart', a.name], check=True)
    print(os.path.join(a.out, a.name))


if __name__ == '__main__':
    main()
