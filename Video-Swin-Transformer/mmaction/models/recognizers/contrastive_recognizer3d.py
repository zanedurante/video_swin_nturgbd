import torch
from torch import nn

from ..builder import RECOGNIZERS, build_loss
from .base import BaseRecognizer
from .recognizer3d import Recognizer3D


@RECOGNIZERS.register_module()
class ContrastiveRecognizer3D(Recognizer3D):
    """3D recognizer model framework."""
    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 contra_loss=None):
        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg)
        if train_cfg is not None and 'feature_extraction' in train_cfg:
            self.feature_extraction = train_cfg['feature_extraction']
        self.set_contra_loss(contra_loss)
    
    def set_contra_loss(self, loss_dict):
        self.contra_loss = build_loss(loss_dict)
    
    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        if self.feature_extraction: # Added to do contrastive learning
            pos_imgs = kwargs['pos_imgs']
            neg_imgs = kwargs['neg_imgs']
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            pos_imgs = pos_imgs.reshape((-1, ) + pos_imgs.shape[2:])
            neg_imgs = neg_imgs.reshape((-1, ) + neg_imgs.shape[2:])
            x = self.extract_feat(imgs) 
            x_pos = self.extract_feat(pos_imgs)
            x_neg = self.extract_feat(neg_imgs)
            contra_loss = self.contra_loss(x, x_pos, x_neg) #(x[0], x[1], x[2]) #(x, x_pos, x_neg)
            return {"contra_loss":contra_loss}
            
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses
    
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        
        #data_batch, pos_batch, neg_batch = data_batch
        
        imgs = data_batch['imgs']
        #pos_imgs = pos_batch['imgs']
        #neg_imgs = neg_batch['imgs']
        
        label = data_batch['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]
        
        losses = self(imgs, label, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs