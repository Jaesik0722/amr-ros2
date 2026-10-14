"""Livox Mid-360 approximation as an RTX OmniLidar prim.

Isaac Sim ships no Livox profile. The Mid-360 uses a non-repetitive pattern; this
approximates it with a rotary scan covering the same field of view and point rate:
360 deg horizontal, -7..+52 deg vertical, 10 Hz, 40 channels x 500 points per
revolution = 200k points/s, 0.1-40 m range.
"""

import numpy as np
from pxr import Sdf

CHANNELS = 40
ELEVATION_DEG = (-7.0, 52.0)
SCAN_RATE_HZ = 10
POINTS_PER_REV = 500  # per channel, 0.72 deg azimuth step
# RTX yields ~0.61 of reportRate/scanRate points per revolution; compensate.
REPORT_RATE_SCALE = 1.63


def create_mid360(stage, path):
    prim = stage.DefinePrim(path, 'OmniLidar')
    prim.AddAppliedSchema('OmniSensorGenericLidarCoreAPI')
    elev = np.linspace(*ELEVATION_DEG, CHANNELS)
    zeros = [0.0] * CHANNELS
    core = 'omni:sensor:Core:'
    es = core + 'emitterState:s001:'
    F, U, Fa, Ua, T = (Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.UInt,
                       Sdf.ValueTypeNames.FloatArray, Sdf.ValueTypeNames.UIntArray,
                       Sdf.ValueTypeNames.Token)
    attrs = {
        es + 'azimuthDeg': (Fa, zeros),
        es + 'elevationDeg': (Fa, [float(e) for e in elev]),
        es + 'channelId': (Ua, list(range(1, CHANNELS + 1))),
        es + 'fireTimeNs': (Ua, [i * 1000 for i in range(CHANNELS)]),
        es + 'distanceCorrectionM': (Fa, zeros),
        es + 'focalDistM': (Fa, zeros),
        es + 'focalSlope': (Fa, zeros),
        es + 'horOffsetM': (Fa, zeros),
        es + 'vertOffsetM': (Fa, zeros),
        es + 'reportRateDiv': (Fa, zeros),
        core + 'numberOfChannels': (U, CHANNELS),
        core + 'numberOfEmitters': (U, CHANNELS),
        core + 'scanType': (T, 'ROTARY'),
        core + 'rotationDirection': (T, 'CW'),
        core + 'rayType': (T, 'IDEALIZED'),
        core + 'scanRateBaseHz': (U, SCAN_RATE_HZ),
        core + 'reportRateBaseHz': (U, int(SCAN_RATE_HZ * POINTS_PER_REV * REPORT_RATE_SCALE)),
        core + 'nearRangeM': (F, 0.1),
        core + 'farRangeM': (F, 40.0),
        core + 'rangeResolutionM': (F, 0.002),
        core + 'rangeAccuracyM': (F, 0.02),
        core + 'avgPowerW': (F, 0.002),
        core + 'minReflectance': (F, 0.1),
        core + 'pulseTimeNs': (U, 6),
        core + 'maxReturns': (U, 1),
        core + 'azimuthErrorMean': (F, 0.0),
        core + 'azimuthErrorStd': (F, 0.0),
        core + 'elevationErrorMean': (F, 0.0),
        core + 'elevationErrorStd': (F, 0.0),
        core + 'startAzimuthOffsetDeg': (F, 0.0),
        core + 'intensityProcessing': (T, 'NORMALIZATION'),
        core + 'intensityMappingType': (T, 'LINEAR'),
        core + 'skipDroppingInvalidPoints': (Sdf.ValueTypeNames.Bool, False),
    }
    for name, (vtype, value) in attrs.items():
        prim.CreateAttribute(name, vtype).Set(value)
    prim.CreateAttribute('omni:sensor:modelName',
                         Sdf.ValueTypeNames.String).Set('Livox_Mid360_approx')
    return prim
