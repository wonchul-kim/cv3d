
In order to estimate 6d pose by **FoundationPose**, there are two methods:

- `model-based`

- `model-free`

However, in both methods, CAD is required for an object to estimate 6d pose.

It means `model-free` requires 3D reconstruction to get CAD and in order to do this, 

I used `apriltag` to get data to train a model, which is to reconstruct 3d shape.

- camera: Intel Realsense

    - color & depth
