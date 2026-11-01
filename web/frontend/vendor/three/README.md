# Three.js 0.170.0

Only the runtime modules used by BodyScan are included. No frontend build is needed.

Unmodified sources from https://unpkg.com/three@0.170.0/:

- `build/three.module.js`
- `examples/jsm/controls/OrbitControls.js`
- `examples/jsm/loaders/OBJLoader.js`
- `LICENSE` (MIT)

The import map in `index.html` resolves `three` to this local copy. Update all three
modules together when changing versions, and verify the viewer before upgrading.
