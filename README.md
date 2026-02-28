# TAPF
Topology-Aware Partial Fine-tuning of ViTs toward Human Behavior Recognition
# Abstract
Human behavior recognition plays significant role in the fields such as intelligent surveillance and human-computer interaction, and has been a research hotspot in computer vision community. Fine-tuning pre-trained models for behavior recognition is currently a prevalent approach. However, the visual encoders employed in existing methods mostly rely on end-to-end parameter estimation, which cannot preserve the original semantics of behaviors. Furthermore, these methods rarely consider the correlation between pre-trained parameters and the behavior recognition tasks during fine-tuning. This paper proposes a Topology-Aware Partial Fine-tuning (TAPF) method for pre-trained model transfer toward human behavior recognition. We adopt Vision Transformer (ViT) as the pre-trained model and enforce that the output token features of ViT preserve the same topological relationships as those among the original image patches. This design preserves the intra-person interactions between different body parts and inter-person interactions between different individuals, thereby enables more effective representation of behavior semantics. Meanwhile, we identify task-relevant parameters based on correlation analysis, freeze these parameters and fully fine-tune the remaining irrelevant parameters to fully unleash the performance of the pre-trained model. Experimental results on multiple datasets demonstrate that our method outperforms existing approaches, indicating that topology-aware semantic representation of behaviors and irrelevant pre-trained parameter fine-tuning can effectively enhance the performance of pre-trained models on human behavior recognition tasks.

### Requirements
- `torch==2.4.1+cu124`
- `torchvision==0.19.1+cu124`
- `numpy==2.0.2`
- `pandas==2.2.3`

### Training
To run the training, use the following command:

```bash
python TAPF.py
