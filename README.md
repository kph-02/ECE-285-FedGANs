# ECE 285 FedGANs - 25 Winter

## Reference paper

This is a study of the paper [Universal Aggregation Federated GANs](https://arxiv.org/pdf/2102.04655). 

## Dataset

We use the HAM10000 (“Human Against Machine with 10,000 dermatoscopic images”) dataset for train, which comprises a diverse collection of 10,015 dermatoscopic images of pigmented skin lesions—spanning seven diagnostic categories (melanoma, nevus, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, and vascular lesions)—captured from multiple clinical centers worldwide

## Metrics

In our evaluation framework, we employ a membership-inference attack (MIA) to quantify the extent to which a trained generator G inadvertently memorizes and leaks information about its training set Dtrain.At its core, the membership-inference metric evaluates a model’s tendency to “memorize” individual training samples by measuring how distinguishable those samples are from unseen data

## Results

We present the empirical evaluation of UAGAN and DCGAN in terms of both generative performance and privacy leakage. The models are assessed using standard generative metrics—Fréchet Inception Distance (FID) and Inception Score (IS)—as well as Membership Inference Attacks (MIA) to measure privacy risk.

### Fréchet Inception Distance (FID):
- DCGAN: 76.686
- UAGAN: 26.085

### Inception Score (IS):
- DCGAN: 1.062 ±0.001
- UAGAN: 1.072 ±0.006

### Membership Inference Attack (MIA) Results

<img width="300" alt="Screenshot 2025-06-29 at 4 06 51 PM" src="https://github.com/user-attachments/assets/701bce53-9005-4289-9116-b9189c35a809" />

### Loss Curves of DCGAN and UAGAN

<img width="400" alt="Screenshot 2025-06-29 at 4 09 01 PM" src="https://github.com/user-attachments/assets/64665ee4-da07-43cd-a84c-b92ad7cd8b9b" />



## Analysis of Results
UAGAN substantially outperforms DCGAN in generative quality. In our experiments, UAGAN reduces the Fréchet Inception Distance from 76.69 (DCGAN) to 26.09—a roughly 66% decrease demonstrating that its outputs lie much closer to the real HAM10000 distribution. Although Inception Scores on medical images remain low overall, UAGAN still achieves a small but meaningful improvement (1.072 ±0.006 vs. 1.062 ±0.001), reflecting better class balance and diversity. Qualitatively, UAGAN generates far richer intra-class variation (e.g., Class 0) and crisper lesion details (e.g., vascular patterns in Class 6), whereas DCGAN exhibits mode collapse and blurrier outputs. 

In terms of privacy, membership inference attacks confirm that UAGAN leaks far less information than DCGAN. DCGAN’s attack AUC of 0.578 and accuracy of 65.6% indicate a strong ability for an adversary to distinguish training samples. By contrast, UAGAN’s attack AUC drops to 0.412 with only 56.3% accuracy—below or near random guessing—showing that sharing only discriminator logits (and never raw images) injects sufficient noise to thwart membership inference.

Together, these findings validate that federated training with Universal Aggregation not only enhances image fidelity and diversity on non-IID medical data but also strengthens privacy guarantees. The dramatic FID improvement confirms better coverage of rare lesion classes, while the reduced MIA performance demonstrates that local discriminators do not overfit to their private subsets, effectively protecting patient data.


### Examples of Generated Images

<img width="552" alt="Screenshot 2025-06-29 at 4 21 07 PM" src="https://github.com/user-attachments/assets/73159cd0-0098-4255-8509-628790bac32e" />






