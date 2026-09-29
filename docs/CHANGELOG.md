# Changelog

## [0.3.0] - 2026-09-17

### Changed

- Migrated to Isaac Sim 6.1: replaced deprecated `isaacsim.core.api`/`isaacsim.core.prims`/`isaacsim.core.utils`
  wrappers with `isaacsim.core.experimental.*`, deprecated `isaacsim.sensors.camera`/`.physics`/`.physx` with
  `isaacsim.sensors.experimental.*`, and rebuilt the DVL sensor on the new `Raycast`/`RaycastSensor` API.
- `LoadButton`/`ResetButton` remain the sole dependency on the deprecated `isaacsim.examples.extension`
  package; no non-deprecated replacement exists yet in 6.1.

## [0.2.0] - 2026-08-()

### Added

- Isaac Sim 5.0 compatibility
- Better ROS2 tutorial and demo
- Standalone scripts for our sea urchin detection ([paper](TODO))

## [0.1.0] - 2025-01-08

### Added

- Initial version of OceanSim Extension
