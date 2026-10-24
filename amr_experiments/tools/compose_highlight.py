"""Compose one experiment's recorded panels into a captioned highlight clip.

  compose_highlight.py OUT_DIR [--speed 3] [--dock front|rear|none] [--no-rviz]

OUT_DIR is what capture_highlight.sh wrote: frames/<camera>/f%05d.jpg, rviz.mp4 and
events.txt. The first line of events.txt is the title, every other line is one of the
runner's own per-trial lines with the second it was printed, so the caption under the
picture always says what the runner was measuring at that moment.

Captions go through drawtext's textfile= instead of text=, because the runner's lines
carry colons, commas and plus signs that drawtext would otherwise parse.
"""

import argparse
import os
import subprocess
import textwrap

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
PANEL_W, PANEL_H = 960, 540
# Characters that still fit across a two-panel frame at the caption size. A runner line
# that says what it measured is longer than that, and drawtext does not wrap: the end of
# the sentence simply runs off the picture, which is where the first mission clip lost
# half of every caption.
CAPTION_COLUMNS = 92


def read_events(path):
    """[(seconds, text)], title first."""
    events = []
    with open(path) as f:
        for line in f:
            seconds, _, text = line.strip().partition(' ')
            try:
                events.append((float(seconds), text))
            except ValueError:
                continue
    return events


def frame_count(directory):
    return len(os.listdir(directory)) if os.path.isdir(directory) else 0


def drawtext(path, size, colour, box, x, y, enable=None):
    out = (f'drawtext=fontfile={FONT}:textfile={path}:fontcolor={colour}:fontsize={size}'
           f':box=1:boxcolor={box}:boxborderw=10:x={x}:y={y}')
    if enable:
        out += f":enable='{enable}'"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--name', default='highlight.mp4')
    ap.add_argument('--speed', type=float, default=3.0)
    ap.add_argument('--dock', default='rear', choices=('front', 'rear', 'both', 'none'),
                    help="'both' puts the two docking cameras side by side in one panel")
    ap.add_argument('--no-rviz', dest='rviz', action='store_false')
    args = ap.parse_args()
    os.chdir(args.out)

    events = read_events('events.txt')
    if not events:
        raise SystemExit('events.txt is empty: the capture did not run')
    title = events[0][1]
    trials = [e for e in events[1:] if e[1] != 'end']
    end = events[-1][0]

    captions = os.path.join('captions')
    os.makedirs(captions, exist_ok=True)
    title_file = os.path.join(captions, 'title.txt')
    with open(title_file, 'w') as f:
        f.write(title)

    # The panels, in the order they are stacked: the two simulator views first, then
    # whatever this experiment had to show (a map needs Nav2, a docking camera needs
    # the ArUco cameras), and black where there is nothing.
    inputs, panels = [], []       # panels: (label, [input index, ...])

    index = 0

    def add_images(name):
        nonlocal index
        inputs.extend(['-framerate', '10', '-i', f'frames/{name}/f%05d.jpg'])
        index += 1
        return index - 1

    def add_video(path):
        nonlocal index
        inputs.extend(['-i', path])
        index += 1
        return index - 1

    for name, label in (('overview', 'Isaac Sim  -  overview'),
                        ('chase', 'Isaac Sim  -  chase camera')):
        if frame_count(f'frames/{name}') == 0:
            raise SystemExit(f'no frames in frames/{name}')
        panels.append((label, [add_images(name)]))
    if args.rviz and os.path.exists('rviz.mp4'):
        panels.append(('RViz  -  SLAM map, Nav2 plan, footprint', [add_video('rviz.mp4')]))
    # 'both' is for the whole-mission clip, which docks to the charger with the front
    # camera and to the cart with the rear one: a single docking panel would be blank
    # for half the mission, so the two share the panel.
    docks = ('front', 'rear') if args.dock == 'both' else (args.dock,)
    docks = [d for d in docks if d != 'none' and frame_count(f'frames/{d}') > 0]
    if docks:
        label = ('Docking cameras  -  front D435 | rear D435' if len(docks) > 1
                 else f'Docking camera  -  {docks[0]} D435')
        panels.append((label, [add_images(d) for d in docks]))

    steps = []
    for i, (label, sources) in enumerate(panels):
        path = os.path.join(captions, f'panel{i}.txt')
        with open(path, 'w') as f:
            f.write(label)
        text = drawtext(path, 22, 'white', 'black@0.55', '12', '12')
        if len(sources) == 1:
            steps.append(f'[{sources[0]}:v]scale={PANEL_W}:{PANEL_H},setpts=PTS-STARTPTS,'
                         f'{text}[p{i}];')
            continue
        width = PANEL_W // len(sources)
        for k, source in enumerate(sources):
            steps.append(f'[{source}:v]scale={width}:{PANEL_H},setpts=PTS-STARTPTS[h{i}{k}];')
        legs = ''.join(f'[h{i}{k}]' for k in range(len(sources)))
        steps.append(f'{legs}hstack=inputs={len(sources)},{text}[p{i}];')

    if len(panels) == 2:
        stack = '[p0][p1]hstack=inputs=2'
        height = PANEL_H
    else:
        for i in range(len(panels), 4):      # black where a panel is missing
            steps.append(f'color=c=black:s={PANEL_W}x{PANEL_H}:r=10,'
                         f'trim=duration={end + 2:.1f}[p{i}];')
        stack = '[p0][p1][p2][p3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0'
        height = 2 * PANEL_H

    overlays = [drawtext(title_file, 30, 'white', 'black@0.8', '(w-text_w)/2',
                         f'{height} - text_h - 18')]
    # A runner prints its trial line when the trial is over, so the caption belongs to
    # the stretch that ends there, not the one that starts there.
    for i, (stop, text) in enumerate(trials):
        start = trials[i - 1][0] if i else events[0][0]
        path = os.path.join(captions, f'trial{i:03d}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(textwrap.wrap(text, CAPTION_COLUMNS)) or text)
        overlays.append(drawtext(path, 26, 'yellow', 'black@0.75', '(w-text_w)/2',
                                 f'{height} - 2 * text_h - 40',
                                 f'between(t,{start:.2f},{stop:.2f})'))
    badge = (f'drawtext=fontfile={FONT}:fontcolor=white:fontsize=26:box=1:'
             f"boxcolor=red@0.75:boxborderw=8:x=w-text_w-16:y=16:text='x{args.speed:g}'")

    graph = (''.join(steps) + stack + ',' + ','.join(overlays) + ',' + badge
             + f',setpts=PTS/{args.speed:g},fps=30[v]')
    command = (['ffmpeg', '-loglevel', 'error', '-y'] + inputs
               + ['-filter_complex', graph, '-map', '[v]',
                  '-t', f'{end / args.speed:.1f}',
                  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                  '-movflags', '+faststart', args.name])
    subprocess.run(command, check=True)
    print(os.path.join(args.out, args.name))


if __name__ == '__main__':
    main()
