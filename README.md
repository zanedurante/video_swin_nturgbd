## Video Swin Repository with Depth and Contrastive Learning Support

In order to complete a course project for CS 231a, a series of adaptations needed to be made to the existing Video-Swin repository so that it could be used for both labeled and unlabeled RGB-D data.  They are summarized below:
* Add config files for fine-tuning Kinetics-400 pre-trained Video-Swin models on NTU RGB-D.
* Add depth modality support for mmaction.
* Add support for contrastive learning within mmaction, which previously only supported supervised learning.
* See the example config files in `configs/` for examples on how to use the updated Video-Swin repo.

If you run into any issues using this code, feel free to create issues and I will try to respond as soon as possible.
