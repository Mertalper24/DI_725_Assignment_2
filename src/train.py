from transformers import DetrForObjectDetection, DetrImageProcessor

id2label = {
    0: "person",
    1: "car",
    2: "truck",
    3: "van",
    4: "motorbike",
    5: "bicycle",
    6: "bus",
    7: "trailer"
}
label2id = {v: k for k, v in id2label.items()}

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(id2label),
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id
)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

print("Model and processor initialized.")
