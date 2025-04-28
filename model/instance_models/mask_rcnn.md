# Mask R-CNN

Mask R-CNN is a powerful two-stage detector that extends Faster R-CNN by adding a branch for predicting segmentation masks. Its architecture works as follows:

1. A backbone network (typically ResNet or ResNeXt with FPN) extracts features from images
2. A Region Proposal Network (RPN) generates region proposals
3. RoIAlign extracts features for each proposal
4. Three parallel heads predict:
   - Class labels
   - Bounding box refinements
   - Binary segmentation masks for each instance

**Strengths:**
- High accuracy for instance segmentation
- Excellent separation of touching or overlapping objects
- Well-established architecture with strong theoretical foundation
- Good handling of objects at different scales through FPN
- Extensive support in major frameworks (PyTorch, TensorFlow)

**Best use cases:** When accuracy is prioritized over speed, and for complex scenes where objects may overlap.
