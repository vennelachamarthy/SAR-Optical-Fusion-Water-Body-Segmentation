SAR-Optical-Fusion-Water-Body-Segmentation

Deep learning pipeline for water body segmentation from satellite imagery, culminating in a dual-encoder transformer model that fuses Sentinel-1 SAR and Sentinel-2 optical data. SAR imagery penetrates cloud cover, while optical imagery provides rich spectral detail — combining the two improves water detection accuracy over either modality alone.

Developed as part of a research internship at NIT Raipur, supervised by Prof. (Dr.) Dilip Singh Sisodia and mentored by Dr. Rikhi Ram Jagat. This repository documents the full progression of the project, from early single-image experiments through to the final fusion architecture.

Project Structure

1. Sentinel-2C Analysis Image
Initial exploration stage. A raw Sentinel-2C tile was acquired and inspected to understand product structure, metadata, and band organization before building any processing pipeline.

2. Sentinel-2C Image Analysis
Early image-processing experiments on the acquired Sentinel-2C tile: generating true-color and false-color composites, and deriving a water mask using spectral index thresholding. This established the baseline logic later reused for automated patch labeling.

3. Sentinel-2C Patch Extraction
The full-scene image and its water mask were split into smaller fixed-size patches, each labeled as `water` or `no\_water` based on pixel water fraction. This patch dataset became the training data for the baseline models below.

4. CNN Model
A convolutional neural network trained as a baseline classifier/segmenter on the extracted patches. Includes training outputs, loss/accuracy curves, and Grad-CAM visualizations showing which regions of each patch the model relied on for its predictions.

5. UNet Model
A UNet architecture trained on the same patch dataset, serving as a stronger baseline than the CNN with encoder-decoder structure and skip connections better suited to pixel-level segmentation. Also includes training curves and Grad-CAM outputs for interpretability.

6. Sentinel-2C Pipeline
An end-to-end single-modality (optical-only) pipeline that combines the full workflow — cropping, water masking, patch generation, and both CNN and UNet inference — into one notebook. This consolidated the earlier standalone experiments (folders 2–5) into a reusable pipeline, and served as the direct precursor to the final fusion model.

S1S2_Fusion_Architecture (final model)
The final and primary contribution of this project: DualSegFormer, a dual-encoder transformer architecture that fuses Sentinel-1 SAR and Sentinel-2 optical imagery for water body segmentation. (To be added once mentor-requested revisions are complete.)

DualSegFormer — Architecture Details
Two separate pretrained MixTransformer (mit-b0) encoders — one processing Sentinel-2 optical input (8 channels, including NDWI/MNDWI), one processing Sentinel-1 SAR input (3 channels: VV, VH, VV−VH)

Cross-modal attention fusion applied at all 4 encoder feature levels, allowing each modality to attend to relevant features from the other
Shared SegFormer-style lightweight MLP decoder producing the final binary water mask ~8.29M parameters total

Training
Dataset: Sen1Floods11 hand-labeled subset (446 chips, 512×512)
Optimizer: AdamW, lr=6e-5, weight decay=1e-4
Loss: BCE + Dice, both masked to exclude no-data pixels
Best validation IoU: \[to be updated after mentor-requested revisions]

Model Checkpoints
Checkpoints are excluded from this repository due to file size. Available via Google Drive:

CNN baseline: \[link]
UNet baseline: \[link]
DualSegFormer (final fusion model): \[link]

Inference
Qualitative inference with the final fusion model was run on two real-world Indian river basin regions using Google Earth Engine imagery:
Mahanadi Basin (Chhattisgarh/Odisha)
Tungabhadra Reservoir (Karnataka)

Notes
Raw satellite imagery (`.tif` files) is excluded from version control; source data can be regenerated via Google Earth Engine.
This repository is structured to show the full development history of the project — from initial single-scene optical experiments to the final SAR-optical fusion model — rather than only the final result.
