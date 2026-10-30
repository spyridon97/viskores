## Add QuadricClusteringDecimation filter

You can now decimate triangle meshes with
`viskores::filter::geometry_refinement::QuadricClusteringDecimation`, a
parallel, deterministic quadric clustering with guarantees on the output.
Set either a target number of points (`SetTargetNumberOfPoints`) or a bound
on the Hausdorff distance to the input (`SetErrorBound`), or both. The
guarantees are bits of `SetGuarantees`. By default (`Topology`,
`NoSelfIntersections`, `SingularCurves`), for an input that is an embedded
surface, closed or with boundary, the output is a surface of the same
topology, without self-intersections, and within the error bound when one is
set. An input that is not a surface, with edges in three or more triangles or
with pinched vertices, is cut into surfaces and glued back with the same
topology (error mode only); its singular curves are decimated too, in
segments that agree across the sheets meeting there, unless the
`SingularCurves` bit is cleared. Without the `Topology` bit, the
geometric-only mode accepts any triangle complex and keeps only the error
bound. The `ErrorOutputToInput` and `ErrorInputToOutput` bits select the sides
of the error bound, both by default. The output is bitwise
identical on every device and thread count.

Two building blocks of the filter are available on their own:

* `viskores/ExactPredicates.h` provides exact orientation tests and exact
  segment-triangle and triangle-triangle intersection tests, with adaptive
  floating-point filters.
* `viskores::worklet::spatialstructure::BoundingVolumeHierarchy` is a linear
  bounding volume hierarchy over boxes in double precision, built in
  parallel, with box-overlap and exact nearest-primitive queries inside
  worklets.
