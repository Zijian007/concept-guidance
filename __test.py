from concept_guidance.data.open_assistant import get_open_assistant_messages
from concept_guidance.data.toxic_completions import get_toxic_completions_messages
# from concept_guidance.data.truthfulqa import get_truthfulqa_messages

# Humor
examples = get_open_assistant_messages(label_key="humor", max_messages=512)

# Creativity
# examples = get_open_assistant_messages(label_key="creativity", max_messages=512)

# Quality
# examples = get_open_assistant_messages(label_key="quality", max_messages=512)

# Compliance
# WARNING: ToxicCompletions contains offensive/harmful user prompts
# examples = get_toxic_completions_messages(max_messages=512)

# Truthfulness
# examples = get_truthfulqa_messages(max_messages=512)
from transformers import AutoModelForCausalLM, AutoTokenizer
from concept_guidance.activations import compute_activations
from concept_guidance.models.difference_in_means import DiMProbe
import torch
device = torch.device("cuda:7")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map = device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Compute model activations
activations, labels = compute_activations(model, tokenizer, examples)

# Train a probe on the activations
probe = DiMProbe()  # or LogisticProbe() or PCAProbe()
probe.fit(activations, labels)

# To get the vectors directly
concept_vectors = probe.get_concept_vectors()

# To save the probe
probe.save("concept.safetensors")