import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temp=0.07):
        super(CLIPLoss, self).__init__()
        self.temp = nn.Parameter(torch.tensor(temp))

    def forward(self, batch_img_emb, batch_obj_emb):
        '''
        input1(batch_img_emb) : (n, dim) tensor
        input2(batch_obj_emb) : (n, dim) tensor
        -----------------------------
        output(loss) : scalar
        '''
        # Normalize embeddings
        normalized_img_emb = F.normalize(batch_img_emb, dim=1, p=2)
        normalized_obj_emb = F.normalize(batch_obj_emb, dim=1, p=2)

        # Compute cosine similarity logits, scale with temp
        logits = torch.matmul(normalized_img_emb, normalized_obj_emb.T) * torch.exp(self.temp)

        # Symmetric loss function (CLIP loss)
        labels = torch.arange(logits.shape[0])
        loss_img = F.cross_entropy(logits, labels, reduction='mean')  # img-to-obj loss
        loss_obj = F.cross_entropy(logits.T, labels, reduction='mean')  # obj-to-img loss
        loss = (loss_img + loss_obj) / 2

        return loss


if __name__ == "__main__":
    clip_loss_fn = CLIPLoss()
    
    batch_img_emb = torch.Tensor([[2, 3, 5],    # img1
                                  [3, 3, 4],    # img2
                                  [4, 1, 5]])   # img3
    batch_obj_emb = torch.Tensor([[2, 3, 5],    # obj1
                                  [3, 3, 4],    # obj2
                                  [4, 1, 5]])   # obj3
    
    loss = clip_loss_fn(batch_img_emb, batch_obj_emb)
    print(loss)
