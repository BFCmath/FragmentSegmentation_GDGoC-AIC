# Cascade Mask R-CNN

Cascade Mask R-CNN improves on Mask R-CNN with a multi-stage detection refinement process:

1. Uses same backbone as Mask R-CNN
2. Implements a cascade of detection heads with increasing IoU thresholds
3. Each stage refines the predictions from the previous stage
4. The final stage produces high-quality instance segmentation masks
5. Trained with progressively more stringent criteria

**Strengths:**
- Higher accuracy than standard Mask R-CNN
- Better handling of high-quality detections
- More precise boundary delineation
- Better performance on challenging instances
- Superior mask quality, especially at object boundaries

**Best use cases:** When highest possible accuracy is required and computational resources are available, such as medical imaging or precise industrial applications.
