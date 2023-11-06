from transformers import ViltProcessor, ViltForImageAndTextRetrieval
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch


class ViLTWrapper:
    def __init__(self, model, processor, device):
        self.model = model
        self.device = device
        self.processor = processor

    def tokenize(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def get_text_embeddings(self, texts):
        inputs = self.tokenize(texts)
        outputs = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        return outputs.last_hidden_state

    def get_image_embeddings(self, images):
        # Assumes images are already preprocessed
        outputs = self.model(images=images)
        return outputs.last_hidden_state

    def get_retrieval_scores(self, texts, images):
        text_embeddings = self.get_text_embeddings(texts)
        image_embeddings = self.get_image_embeddings(images)
        scores = F.cosine_similarity(text_embeddings, image_embeddings)
        return scores

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")

        scores = []
        # Iterate over the batched data
        for batch in tqdm_loader:
            batch_scores = []
            i_options = batch["image_options"][0]
            i_options = i_options.to(self.device)
            c_options0 = batch["caption_options"][0]
            c_options1 = batch["caption_options"][1]
            for i in range(len(i_options)):
                inputs0 = self.processor(
                    images=i_options[i],
                    text=c_options0[i],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs1 = self.processor(
                    images=i_options[i],
                    text=c_options1[i],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                inputs0 = inputs0.to(self.device)
                inputs1 = inputs1.to(self.device)

                outputs0 = self.model(**inputs0)
                outputs1 = self.model(**inputs1)

                score0 = outputs0.logits[:, 0].item()
                score1 = outputs1.logits[:, 0].item()

                batch_scores.append([score0, score1])  # B x L (16x2)

            batch_scores = np.expand_dims(batch_scores, axis=1)  # B x K x L (16x1x2)

            scores.append(batch_scores)

        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        return all_scores
