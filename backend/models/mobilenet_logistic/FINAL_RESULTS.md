# Final experiment output

Method: Frozen ImageNet-pretrained MobileNetV3-small +
temporal mean/std pooling + StandardScaler + logistic regression.

Input: 16 uniformly sampled RGB frames, resized to 160 x 160,
with ImageNet normalization.
Features: 576 temporal means + 576 temporal standard deviations.
Selected C: 0.01
Classification threshold: 0.5

Dataset split:
- Training: 1,440 clips
- Validation: 160 clips
- Test: 400 clips

Test results:
- Accuracy: 76.00%
- Precision: 78.89%
- Recall: 71.00%
- F1: 74.74%
- Loss: 0.5188

Class labels: 0 = NonFight; 1 = Fight.
The classifier file includes the fitted StandardScaler.
The extractor file contains torchvision MobileNetV3-small
.features weights, not the LSTM model.
Exact split and feature settings: feature_metadata.json.

These results apply to this dataset experiment.
Dashboard integration has not been completed.
