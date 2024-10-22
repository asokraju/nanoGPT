import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from typing import Optional
import logging
from torch.utils.tensorboard import SummaryWriter
from model_hf import GPTConfig, GPTLMHeadModel, MoEUsageLoggingCallback

# Define a small synthetic dataset for debugging
class SyntheticDataset(Dataset):
    def __init__(self, vocab_size: int, block_size: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.block_size,))
        labels = input_ids.clone()  # For simplicity, labels are the same as input_ids
        return {'input_ids': input_ids, 'labels': labels}



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MoELogger')

# Prepare the model configuration
config = GPTConfig(
    block_size=16,  # Small block size for debugging
    vocab_size=100,  # Small vocab size for debugging
    n_layer=2,      # Few layers for faster training
    n_head=2,
    n_embd=32,
    use_moe=True,  # Enable Mixture of Experts
    num_experts=4,
    num_experts_per_tok=2,
    moe_loss=True,
    moe_loss_type = "entropy_regularization",  # Type of load balancing loss "variance_penalty", "entropy_regularization", "diversity_regularization"
    moe_loss_coef = 1e0, 
)

# Initialize the model
resume = True
ckpt_path='results\checkpoint-500'
if resume:
    model = GPTLMHeadModel.from_pretrained(pretrained_model_name_or_path=ckpt_path,
                                           local_files_only=True,                                           device_map='cuda')
else:
    model = GPTLMHeadModel(config)

# Prepare the synthetic dataset
dataset = SyntheticDataset(vocab_size=config.vocab_size, block_size=config.block_size)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to=["tensorboard"],
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # callbacks=[moe_logging_callback],
)
moe_logging_callback = MoEUsageLoggingCallback(trainer=trainer, logger=logger, log_interval=10, log_dir='./logs/moe_logs')
trainer.add_callback(moe_logging_callback)

# Train the model
if resume:
    trainer.train(resume_from_checkpoint=ckpt_path)
else:
    trainer.train()