"""Write the docking ArUco markers as PNGs (for printing and for the simulator).

  python3 make_markers.py ../config/docking_targets.yaml ../markers

Each image is the marker (black square incl. its 1-cell border) with a 1-cell white
quiet zone, so it is 8/6 of the configured marker_size. Print at that size.
"""
import sys

import cv2
import numpy as np
import yaml

CELL = 50  # pixels per marker cell


def main():
    cfg_path, out_dir = sys.argv[1:3]
    cfg = yaml.safe_load(open(cfg_path))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['aruco']['dictionary']))
    size_m = cfg['aruco']['marker_size']
    for target in ('charger', 'cart'):
        for marker_id in cfg[target]['markers']:
            if hasattr(cv2.aruco, 'generateImageMarker'):
                img = cv2.aruco.generateImageMarker(dictionary, int(marker_id), 6 * CELL)
            else:
                img = cv2.aruco.drawMarker(dictionary, int(marker_id), 6 * CELL)
            img = np.pad(img, CELL, constant_values=255)
            path = f'{out_dir}/aruco_{marker_id}.png'
            cv2.imwrite(path, img)
            print(f'{path}: {target}, print {size_m * 8 / 6 * 1000:.0f} mm wide '
                  f'(black square {size_m * 1000:.0f} mm)')


if __name__ == '__main__':
    main()
