import torch
import torch.nn.functional as F
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False, topk=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.topk = topk
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits = logits.detach().to(self.correct.device)
        target = target.detach().to(self.correct.device)

        logits = logits[target!=-100]
        target = target[target!=-100]
        if target.numel() == 0:
            return 1

        if len(logits.size()) == 2:
            if self.topk == 1:
                preds = logits.argmax(dim=-1)
                acc = torch.sum(preds==target)
            else:
                preds = logits.topk(k=self.topk)[1]
                acc = (preds==target.unsqueeze(1)).any(dim=1).sum()
        elif len(logits.size()) == 1:
            acc = torch.sum(logits == target)
        else:
            raise TypeError("Invalid Preds Shape")

        self.correct += acc
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total
    
class IoU(Metric):
    def __init__(self, dist_sync_on_step=False, topk=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.topk = topk
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_3", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_5", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, start_logits, end_logits, start_target, end_target):
        
        if start_target.numel() == 0 or end_target.numel() == 0:
            return 1, 1, 1
        self.total += start_target.numel()
        
        if type(start_logits) == list:
            start_preds = start_logits
            end_preds = end_logits
        else:
            start_logits = start_logits.detach().to(self.correct.device)
            end_logits = end_logits.detach().to(self.correct.device)
            start_target = start_target.detach().to(self.correct.device)
            end_target = end_target.detach().to(self.correct.device)

            start_logits = start_logits[start_target!=-100]
            start_target = start_target[start_target!=-100]
            end_logits = end_logits[end_target!=-100]
            end_target = end_target[end_target!=-100]
            
            # batch_size = start_logits.size(0)
            # logits = torch.cat([start_logits, end_logits], dim=0)
            # prob = F.gumbel_softmax(logits, tau=1.0, dim=1)
            # lb = torch.zeros(prob.shape).to(prob.device)
            # prob = torch.where(prob<0.85, lb, prob)
            # index = prob.argmax(dim=1)
            # gumbel_start_preds = index[:batch_size]
            # gumbel_end_preds = index[batch_size:]
            # start_preds = gumbel_start_preds
            # end_preds = gumbel_end_preds
            
            
            start_preds = start_logits.argmax(dim=1).tolist()
            end_preds = end_logits.argmax(dim=1).tolist()
            
        start_target = start_target.tolist()
        end_target = end_target.tolist()
        

        for i in range(len(start_target)):
            if start_preds[i] == start_target[i] and end_preds[i] == end_target[i]:
                self.correct += 1.
                self.correct_3 += 1
                self.correct_5 += 1
            elif start_preds[i] >= end_target[i] or end_preds[i] <= start_target[i]:
                self.correct += 0.
                self.correct_3 += 0
                self.correct_5 += 0
            else:
                ll = min(start_preds[i], start_target[i])
                lr = max(start_preds[i], start_target[i])
                rr = max(end_preds[i], end_target[i])
                rl = min(end_preds[i], end_target[i])
                iou = max((rl-lr+1), 0)/(rr-ll+1)
                self.correct += iou
                
                if iou >= 0.3:
                    self.correct_3 += 1
                if iou >= 0.5:
                    self.correct_5 += 1
                    
    def compute(self):
        return self.correct / self.total, self.correct_3 / self.total, self.correct_5 / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total

def rouge_n(gold, pred, ignore=[',', '.']):
    if type(gold) is list:
        rouges = []
        for g, p in zip(gold, pred):
            rouge_all = 0
            hit_n = 0
            if ignore is None:
                for token in g:
                    if token in p:
                        hit_n += 1
                return hit_n / len(g)
            else:
                sum_len = 0
                for token in g:
                    if token in ignore:
                        continue
                    if token in p:
                        hit_n += 1
                    sum_len += 1
                if sum_len:
                    rouge_all += hit_n / sum_len
            if len(gold) > 0:
                rouge_all /= len(gold)
            rouges.append(rouge_all)
        return rouges  
    
    else:
        hit_n = 0
        if ignore is None:
            for token in gold:
                if token in pred:
                    hit_n += 1
            return hit_n / len(gold)
        else:
            sum_len = 0
            for token in gold:
                if token in ignore:
                    continue
                if token in pred:
                    hit_n += 1
                sum_len += 1
            if sum_len:
                return hit_n / sum_len
            else:
                return 0