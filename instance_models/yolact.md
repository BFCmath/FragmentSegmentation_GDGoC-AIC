# YOLACT (You Only Look At CoefficienTs)

YOLACT is a single-shot instance segmentation model designed for real-time performance. It works by:

1. Using a backbone (typically ResNet + FPN) to extract features
2. Generating a set of prototype masks (shared across all instances)
3. Predicting per-instance mask coefficients along with boxes and class scores
4. Linearly combining prototypes using coefficients to generate instance masks
5. Applying Non-Maximum Suppression to remove duplicates

**Strengths:**
- Significantly faster than Mask R-CNN (up to 33 FPS)
- Single-stage design simplifies implementation
- End-to-end trainable architecture
- Good balance between speed and accuracy

**Best use cases:** Real-time applications where speed is crucial, like video analysis or interactive systems.
