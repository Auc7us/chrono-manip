These demo-local CUDA sources adapt the installed Project Chrono Sensor shaders.
The upstream copyright notices and BSD license are retained in `LICENSE.chrono`.

`camera_raygen.cu` adds the photograph-fitted equidistant polynomial to the FOV
lens branch. Pinhole/radial rays keep their normal behavior. The material shader
uses the local `ChOptixSpotLight.cuh`: special internal ranges -2 and -3 represent
finite floor/ceiling diffuse cells. Their near-field denominator is d² + A/π,
the on-axis irradiance factor for a uniform radiance disk of area A. The other
lights retain the installed inverse-square attenuation. This is a bounded
diffuse-cell approximation, rather than a complete area-source transport model.

`reference_camera.py` compiles and selects a private PTX directory for this demo
process. It does not modify the Chrono checkout or installation. The current
container's RTX 4080 / CUDA/OptiX headers are used for the first cached build.
