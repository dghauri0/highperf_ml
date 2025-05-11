import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import wandb
import os
import time
import random
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, ViTFeatureExtractor, get_cosine_schedule_with_warmup
#from dataset import MSCOCODataset
import sys
import io
import contextlib
import math
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image

class MSCOCODataset(Dataset):
    def __init__(self, annotations_path, images_dir, image_processor, tokenizer, reduce_fraction=0.55):

        # Load MS COCO annotations from JSON
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # Use and reduce the list of annotations
        annotations = self.coco_data['annotations']
        if reduce_fraction < 1.0:
            random.seed(42)
            annotations = random.sample(annotations, int(len(annotations) * reduce_fraction))
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        caption = ann['caption']
        image_id = ann['image_id']
        image_path = os.path.join(self.images_dir, f"{str(image_id).zfill(12)}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None
        image_tensor = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        text_inputs = self.tokenizer(caption, return_tensors="pt", padding=False, truncation=True)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        return {
            "image_inputs": image_tensor,
            "text_inputs": text_inputs,
            "caption": caption,
            "image_id": image_id
        }

class CrossModalModel(nn.Module):
    def __init__(self, image_model_name="google/vit-base-patch16-224", text_model_name="bert-base-uncased", embed_dim=256):
        super(CrossModalModel, self).__init__()
        from transformers import ViTModel, BertModel
        self.image_encoder = ViTModel.from_pretrained(image_model_name)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, image_inputs, text_inputs):
        image_outputs = self.image_encoder(pixel_values=image_inputs)
        text_outputs = self.text_encoder(**text_inputs)
        image_cls = image_outputs.last_hidden_state[:, 0, :]
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        image_embeds = F.normalize(self.image_proj(image_cls), p=2, dim=-1)
        text_embeds = F.normalize(self.text_proj(text_cls), p=2, dim=-1)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits = logit_scale * (image_embeds @ text_embeds.T)
        return logits

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def compute_contrastive_loss(logits, temperature=0.07):
    targets = torch.arange(logits.size(0)).to(logits.device)
    logits = logits / temperature
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2

def compute_recall(logits, topk=(1, 5)):
    targets = torch.arange(logits.size(0)).to(logits.device)
    recall_scores = {}
    for k in topk:
        _, indices = logits.topk(k, dim=1)
        recall_k = (indices == targets.unsqueeze(1)).float().sum().item() / targets.size(0)
        recall_scores[f"recall@{k}"] = recall_k
    return recall_scores

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print("[collate_fn] Empty batch received.")
        return None
    try:
        image_inputs = torch.stack([item["image_inputs"] for item in batch])
        input_id_list = [item["text_inputs"]["input_ids"] for item in batch]
        mask_list = [item["text_inputs"]["attention_mask"] for item in batch]
        padded_input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(mask_list, batch_first=True, padding_value=0)
        captions = [item["caption"] for item in batch]
        return image_inputs, padded_input_ids, padded_masks, captions
    except Exception as e:
        print(f"[collate_fn] Exception during collation: {e}")
        return None

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project="hpml-project")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    annotations_path_train = "/scratch/dg4140/project/coco/annotations/captions_train2017.json"
    images_dir_train = "/scratch/dg4140/project/coco/images/train2017"

    annotations_path_val = "/scratch/dg4140/project/coco/annotations/captions_val2017.json"
    images_dir_val = "/scratch/dg4140/project/coco/images/val2017"

    #annotations_path_train = "/scratch/ac11274/project/annotations/captions_train2017.json"
    #images_dir_train = "/scratch/ac11274/project/train2017"
    #annotations_path_val = "/scratch/ac11274/project/annotations/captions_val2017.json"
    #images_dir_val = "/scratch/ac11274/project/val2017"

    train_dataset = MSCOCODataset(annotations_path_train, images_dir_train, processor, tokenizer, reduce_fraction=0.45)
    val_dataset = MSCOCODataset(annotations_path_val, images_dir_val, processor, tokenizer, reduce_fraction=0.45)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, sampler=val_sampler, num_workers=4, collate_fn=collate_fn)

    model = CrossModalModel().cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=len(train_loader) * 7)
    temperature = 0.07

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for epoch in range(4):
            model.train()
            train_sampler.set_epoch(epoch)
            for i, batch in enumerate(train_loader):
                if batch is None:
                    print(f"[rank{rank}] Skipped empty batch {i}")
                    continue
                image_inputs, input_ids, attention_mask, _ = batch
                image_inputs = image_inputs.cuda(rank, non_blocking=True)
                input_ids = input_ids.cuda(rank, non_blocking=True)
                attention_mask = attention_mask.cuda(rank, non_blocking=True)

                optimizer.zero_grad()
                with record_function("model_forward"):
                    logits = model(image_inputs=image_inputs, text_inputs={"input_ids": input_ids, "attention_mask": attention_mask})
                    loss = compute_contrastive_loss(logits, temperature)
                with record_function("model_backward"):
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                if rank == 0 and i % 10 == 0:
                    wandb.log({"train_loss": loss.item(), "epoch": epoch})

            model.eval()
            with torch.no_grad():
                val_losses = []
                all_metrics = []
                for i, batch in enumerate(val_loader):
                    if batch is None:
                        print(f"[rank{rank}] Skipped empty val batch {i}")
                        continue
                    image_inputs, input_ids, attention_mask, _ = batch
                    image_inputs = image_inputs.cuda(rank, non_blocking=True)
                    input_ids = input_ids.cuda(rank, non_blocking=True)
                    attention_mask = attention_mask.cuda(rank, non_blocking=True)

                    logits = model(image_inputs=image_inputs, text_inputs={"input_ids": input_ids, "attention_mask": attention_mask})
                    val_loss = compute_contrastive_loss(logits, temperature)
                    val_losses.append(val_loss.item())
                    #all_metrics.append(compute_recall(logits))
                    clip_scores = cosine_similarity(model.module.image_proj(image_inputs), model.module.text_proj(input_ids), dim=-1)
                    avg_clip_score = clip_scores.mean().item()
                    all_metrics.append({**compute_recall(logits), "clip_score": avg_clip_score})

                if rank == 0:
                    if val_losses:
                        avg_val_loss = sum(val_losses) / len(val_losses)
                        avg_metrics = {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in all_metrics[0]}
                        wandb.log({"val_loss": avg_val_loss, **avg_metrics, "epoch": epoch})
                    else:
                        print("[rank0] Warning: No validation loss recorded. Possibly all batches were skipped.")

        if rank == 0:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    if rank == 0:
        torch.save(model.module.state_dict(), "clip_final_model.pth")
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[rank{os.environ.get('LOCAL_RANK')}] Exception:")
        traceback.print_exc()
        raise
