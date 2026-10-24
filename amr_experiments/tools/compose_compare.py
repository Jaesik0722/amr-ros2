"""Put two recorded runs side by side: the same experiment, the two cart constructions.

  compose_compare.py OUT.mp4 --title "..." --note "..." \
      --left DIR  --left-label  "single-axle trailer" --left-start 62.9 \
      --right DIR --right-label "four-caster trolley" --right-start 0.0 \
      --seconds 105 --cameras overview,chase --speed 3

Each DIR is a capture directory written by capture_highlight.sh: frames/<camera>/f%05d.jpg
at 10 Hz, contiguous from f00000. The two runs were recorded at different times, so each
side gets its own start offset, taken from the trial line in that run's events.txt; what
lines up in the finished clip is the part of the route, not the wall clock.

The grid is one column per cart and one row per camera, so the same moment of the same
route is read across. A clip is one chapter; several are joined with
`ffmpeg -f concat`, which is what record_compare.sh does.
"""

import argparse
import os
import subprocess

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
PANEL_W, PANEL_H = 960, 540


def drawtext(path, size, colour, box, x, y):
    return (f'drawtext=fontfile={FONT}:textfile={path}:fontcolor={colour}:fontsize={size}'
            f':box=1:boxcolor={box}:boxborderw=10:x={x}:y={y}')


def frame_count(directory):
    return len(os.listdir(directory)) if os.path.isdir(directory) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--left', required=True)
    ap.add_argument('--right', required=True)
    ap.add_argument('--left-label', default='left')
    ap.add_argument('--right-label', default='right')
    ap.add_argument('--left-start', type=float, default=0.0)
    ap.add_argument('--right-start', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, required=True)
    ap.add_argument('--cameras', default='overview,chase')
    ap.add_argument('--speed', type=float, default=3.0)
    ap.add_argument('--title', default='')
    ap.add_argument('--note', default='')
    ap.add_argument('--captions', default=None,
                    help='directory for the caption text files (default: beside OUT)')
    args = ap.parse_args()

    cameras = args.cameras.split(',')
    sides = [(args.left, args.left_label, args.left_start),
             (args.right, args.right_label, args.right_start)]
    captions = args.captions or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                             'captions')
    os.makedirs(captions, exist_ok=True)

    # The grid is filled row by row: camera 0 left and right, then camera 1, so xstack's
    # input order has to be that, not all of one side first.
    inputs, steps, cells = [], [], []
    frames = int(round(args.seconds * 10))
    for row, camera in enumerate(cameras):
        for column, (directory, label, start) in enumerate(sides):
            path = os.path.join(directory, 'frames', camera)
            available = frame_count(path)
            first = int(round(start * 10))
            if available <= first:
                raise SystemExit(f'{path}: {available} frames, needs more than {first}')
            if available - first < frames:
                raise SystemExit(f'{path}: only {(available - first) / 10:.1f} s after the '
                                 f'offset, asked for {args.seconds:.1f} s')
            inputs += ['-framerate', '10', '-start_number', str(first),
                       '-i', os.path.join(path, 'f%05d.jpg')]
            index = len(cells)
            text = os.path.join(captions, f'cell{row}{column}.txt')
            with open(text, 'w') as f:
                f.write(f'{label}  -  {camera}')
            steps.append(f'[{index}:v]scale={PANEL_W}:{PANEL_H},setpts=PTS-STARTPTS,'
                         f'trim=duration={args.seconds:.2f},'
                         f'{drawtext(text, 24, "white", "black@0.6", "12", "12")}[c{index}];')
            cells.append(f'[c{index}]')

    height = PANEL_H * len(cameras)
    if len(cells) == 2:
        stack = f'{"".join(cells)}hstack=inputs=2'
    else:
        # xstack places each input by expression. The x of a cell is the width of the
        # cell to its left in the same row, the y the sum of the heights above it in the
        # same column, both named by input index.
        places = []
        for r in range(len(cameras)):
            for c in range(2):
                x = '0' if c == 0 else f'w{r * 2}'
                y = '0' if r == 0 else '+'.join(f'h{k * 2 + c}' for k in range(r))
                places.append(f'{x}_{y}')
        layout = '|'.join(places)
        stack = f'{"".join(cells)}xstack=inputs={len(cells)}:layout={layout}'

    overlays = []
    if args.title:
        path = os.path.join(captions, 'title.txt')
        with open(path, 'w') as f:
            f.write(args.title)
        overlays.append(drawtext(path, 32, 'white', 'black@0.8', '(w-text_w)/2',
                                 f'{height} - text_h - 18'))
    if args.note:
        path = os.path.join(captions, 'note.txt')
        with open(path, 'w') as f:
            f.write(args.note)
        overlays.append(drawtext(path, 26, 'yellow', 'black@0.75', '(w-text_w)/2',
                                 f'{height} - 2 * text_h - 40'))
    badge = (f'drawtext=fontfile={FONT}:fontcolor=white:fontsize=26:box=1:'
             f"boxcolor=red@0.75:boxborderw=8:x=w-text_w-16:y=16:text='x{args.speed:g}'")

    graph = (''.join(steps) + stack + ','
             + ''.join(f'{o},' for o in overlays) + badge
             + f',setpts=PTS/{args.speed:g},fps=30[v]')
    command = (['ffmpeg', '-loglevel', 'error', '-y'] + inputs
               + ['-filter_complex', graph, '-map', '[v]',
                  '-t', f'{args.seconds / args.speed:.2f}',
                  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                  '-movflags', '+faststart', args.out])
    subprocess.run(command, check=True)
    print(os.path.abspath(args.out))


if __name__ == '__main__':
    main()
