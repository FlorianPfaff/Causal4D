# Registered-design power and calibration-fragility audit

This is a source-only design diagnostic. It does not change the frozen method, threshold, exclusions, or analysis.

## Session-level precision

| Sessions | 95% half-width (session SD) | d for 80% power | d for 90% power |
| ---: | ---: | ---: | ---: |
| 12 | 0.635 | 0.889 | 1.029 |
| 18 | 0.497 | 0.701 | 0.811 |

## Calibration boundary

The registered 90% execution-block threshold with nine calibration sessions is rank 9 of 9, i.e. the sample maximum. Removing any one calibration session leaves only eight units, for which the formal 90% threshold is not finite without the registered infinite sentinel.
