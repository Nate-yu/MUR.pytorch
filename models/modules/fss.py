import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.distributed as dist
from tqdm import tqdm

class FSS(nn.Module):
    def __init__(self, inplanes, instrides, structure):
        super(FSS, self).__init__()
        self.inplanes = inplanes
        self.instrides = instrides
        self.structure = structure
        self.indexes = nn.ParameterDict()

        for block in self.structure:
            for layer in block['layers']:
                self.indexes["{}_{}".format(block['name'], layer['idx'])] = nn.Parameter(torch.zeros(layer['planes']).long(), requires_grad=False)
                self.add_module("{}_{}_upsample".format(block['name'], layer['idx']),
                                nn.UpsamplingBilinear2d(scale_factor=self.instrides[layer['idx']] / block['stride']))

    def forward(self, feats):
        block_feats = {}
        for block in self.structure:
            block_feats[block['name']] = []
            ref_size = None  # 用于记录基准尺寸
            for layer in block['layers']:
                feat_c = torch.index_select(feats[layer['idx']], 1, self.indexes["{}_{}".format(block['name'], layer['idx'])].data)
                feat_c = getattr(self, "{}_{}_upsample".format(block['name'], layer['idx']))(feat_c)
                if ref_size is None:
                    ref_size = feat_c.shape[-2:]  # 设置基准尺寸为第一个特征图的尺寸
                else:
                    # 调整当前特征图的尺寸以匹配基准尺寸
                    feat_c = F.interpolate(feat_c, size=ref_size, mode='bilinear', align_corners=False)
                block_feats[block['name']].append(feat_c)
            block_feats[block['name']] = torch.cat(block_feats[block['name']], dim=1)
        return block_feats

    def get_outplanes(self):
        return {block['name']: sum([layer['planes'] for layer in block['layers']]) for block in self.structure}

    def get_outstrides(self):
        return {block['name']: block['stride'] for block in self.structure}

    @torch.no_grad()
    def init_idxs(self, model, train_loader, distributed=True):
        criterion = nn.MSELoss(reduction='none').to(model.device)
        for block in self.structure:
            self.init_block_idxs(block, model, train_loader, criterion, distributed=distributed)

    def init_block_idxs(self, block, model, train_loader, criterion, distributed=True):
        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name'])) if rank == 0 else range(self.init_bsn)
        else:
            tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))

        cri_sum_vec = [torch.zeros(self.inplanes[layer['idx']]).to(model.device) for layer in block['layers']]
        iterator = iter(train_loader)

        for bs_i in tq:
            try:
                input = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                input = next(iterator)

            bb_feats = model.backbone(input)
            feats = bb_feats['feats']

            for i, layer in enumerate(block['layers']):
                layer_name = layer['idx']
                layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(feats[layer_name])
                mse_loss = torch.mean(criterion(layer_feat, layer_feat), dim=1)

                if distributed:
                    mse_loss_list = [mse_loss for _ in range(world_size)]
                    dist.all_gather(mse_loss_list, mse_loss)
                    mse_loss = torch.mean(torch.stack(mse_loss_list, dim=0), dim=0, keepdim=False)

                cri_sum_vec[i] += mse_loss

        for i in range(len(cri_sum_vec)):
            cri_sum_vec[i][torch.isnan(cri_sum_vec[i])] = torch.max(cri_sum_vec[i][~torch.isnan(cri_sum_vec[i])])
            values, indices = torch.topk(cri_sum_vec[i], k=block['layers'][i]['planes'], dim=-1, largest=False)
            values, _ = torch.sort(indices)

            if distributed:
                tensor_list = [values for _ in range(world_size)]
                dist.all_gather(tensor_list, values)
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(tensor_list[0].long())
            else:
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(values.long())
