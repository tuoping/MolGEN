from mdgen.parsing import parse_train_args
args = parse_train_args()
from mdgen.logger import get_logger
logger = get_logger(__name__)

import torch, os
from mdgen.dataset import EquivariantTransformerDataset_Alanine_Dipeptide
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import pytorch_lightning as pl

class ResetLrCallback(pl.Callback):
    def __init__(self, new_lr: float):
        self.new_lr = new_lr

    # runs right after checkpoint restore, before the first batch
    def on_train_epoch_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for pg in optimizer.param_groups:
                pg["lr"] = self.new_lr
        ## (optional) reset schedulers if you wish
        # scheduler = pl_module.lr_schedulers()
        # scheduler.base_lrs = [self.new_lr]
        # scheduler.last_epoch = 1499  # starts fresh


torch.set_float32_matmul_precision('medium')
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset

train_dataset = EquivariantTransformerDataset_Alanine_Dipeptide(args.data_dir, num_species=4, sim_condition=False, tps_condition=False, stage="traj_s1_train")

if args.overfit:
    val_dataset = train_dataset
else:
    val_dataset = EquivariantTransformerDataset_Alanine_Dipeptide(args.data_dir, num_species=4, sim_condition=False, tps_condition=False, stage="traj_s1_val")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

model = EquivariantMDGenWrapper(args)
if args.ckpt is not None:
    checkpoint = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

callbacks_fn = [
    ModelCheckpoint(
        dirpath=os.environ["MODEL_DIR"], 
        # filename="{epoch:03d}-{step:07d}-{val_loss:.4f}",
        # monitor="val_loss",
        save_top_k=-1,
        every_n_epochs=args.ckpt_freq,
        # save_last=True
    ),
    ModelCheckpoint(
        dirpath=os.environ["MODEL_DIR"], 
        filename="{epoch:03d}-{step:07d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        # save_last=True
    ),
    ModelSummary(max_depth=2),
]

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
    num_sanity_val_steps=0,
    precision=args.precision,
    enable_progress_bar=not args.wandb or os.getlogin() == 'hstark',
    gradient_clip_val=args.grad_clip,
    default_root_dir=os.environ["MODEL_DIR"], 
    callbacks=callbacks_fn,
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_freq,
    check_val_every_n_epoch=args.val_epoch_freq,
    logger=False
)


if args.validate:
    # trainer.validate(model, val_loader, ckpt_path=args.ckpt)
    trainer.validate(model, val_loader)
else:
    # trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
    trainer.fit(model, train_loader, val_loader)
