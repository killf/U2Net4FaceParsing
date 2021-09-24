import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime
import os

from utils import *
from optimizers import Optimizers
from datasets import DATASETS
from models import MODELS
from losses import LOSSES
from config import Config


class Solver:
    def __init__(self, cfg: Config):
        self.device = torch.device(cfg.device)
        self.output_dir = cfg.output_dir

        train_data = DATASETS[cfg.dataset](cfg.data_dir, file_list="train.txt", image_size=cfg.image_size, transform=cfg.train_transform)
        self.train_loader = DataLoader(train_data, cfg.batch_size, True, num_workers=4)

        val_data = DATASETS[cfg.dataset](cfg.data_dir, file_list="val.txt", image_size=cfg.image_size, transform=cfg.val_transform)
        self.val_loader = DataLoader(val_data, cfg.batch_size, False)

        self.cfg = cfg.build(len(self.train_loader), train_data.num_classes)
        self.model = MODELS[cfg.model_name](**cfg.model_args).to(self.device)

        self.loss = LOSSES[cfg.loss_name](cfg, **cfg.loss_args).to(self.device)
        self.optimizer = Optimizers[cfg.optimizer_name](self.model, **cfg.optimizer_args)

        self.start_epoch = 1
        self.epochs = cfg.epochs
        self.output_dir = cfg.output_dir
        self.log_dir = os.path.join(self.output_dir, "log")
        self.log_file = os.path.join(self.output_dir, "log.txt")
        self.logger = SummaryWriter(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.best_miou = 0
        self.global_step = 0
        if self.cfg.num_device > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.load_checkpoint()

    def train(self):
        self.adjust_lr(self.cfg.lr)
        for epoch in range(self.start_epoch, self.epochs + 1):
            if epoch in self.cfg.milestones:
                self.adjust_lr(self.cfg.milestones[epoch])
            self.adjust_lr_manual()

            self.train_epoch(epoch)

            val_miou = self.val_epoch() if self.cfg.do_val else -1
            self.save_checkpoint(epoch, val_miou, val_miou >= self.best_miou)

            self.sample(epoch=epoch)
            print()

    def train_epoch(self, epoch):
        self.model.train()

        t, c = Timer(), Counter()
        t.start()
        for step, (img, mask) in enumerate(self.train_loader):
            img, mask = img.to(self.device), mask.to(self.device)
            reader_time = t.elapsed_time()

            d0, d1, d2, d3, d4, d5, d6 = self.model(img)
            loss, losses = self.loss(mask, d0, d1, d2, d3, d4, d5, d6)

            pred = torch.argmax(d0, 1)
            acc = torch.eq(pred, mask).float().mean()
            miou = mean_iou(pred, mask, self.cfg.num_classes)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, losses = float(loss),  [float(item) for item in losses]
            acc, miou = float(acc) * 100, float(miou) * 100
            batch_time = t.elapsed_time()
            c.append(loss=loss, acc=acc, miou=miou, reader_time=reader_time, batch_time=batch_time)
            eta = calculate_eta(len(self.train_loader) - step, c.batch_time)
            self.log(f"[{datetime.datetime.now():%m-%d %H:%M:%S}] "
                     f"[epoch={epoch}/{self.epochs}] "
                     f"step={step + 1}/{len(self.train_loader)} "
                     f"lr={self.optimizer.get_lr():.6f} "
                     f"loss={loss:.4f}/{c.loss:.4f} "
                     f"losses=[{losses[0]:.4f},{losses[1]:.4f},{losses[2]:.4f},{losses[3]:.4f},{losses[4]:.4f},{losses[5]:.4f},{losses[6]:.4f}] "
                     f"acc={acc:.2f}/{c.acc:.2f} "
                     f"miou={miou:.2f}/{c.miou:.2f} "
                     f"batch_time={c.batch_time:.4f}+{c.reader_time:.4f} "
                     f"| ETA {eta}",
                     end="\r",
                     to_file=step % 10 == 0)

            self.logger.add_scalar("train/loss", float(loss), global_step=self.global_step)
            self.logger.add_scalar("train/acc", acc, global_step=self.global_step)
            self.logger.add_scalar("train/miou", miou, global_step=self.global_step)
            self.global_step += 1
            self.logger.flush()

            t.restart()
        print()

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()

        c = Counter()
        for step, (img, mask) in enumerate(self.val_loader):
            img, mask = img.to(self.device), mask.to(self.device)

            pred = self.model(img)[0]

            pred = torch.argmax(pred, 1)
            acc = torch.eq(pred, mask).float().mean()
            miou = mean_iou(pred, mask, self.cfg.num_classes)

            c.append(acc=float(acc) * 100, miou=float(miou) * 100)
            print(f"[VAL] {step + 1}/{len(self.val_loader)}", end="\r", flush=True)

        self.log(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                 f"[VAL] "
                 f"acc={c.acc:.2f} "
                 f"miou={c.miou:.2f}")

        self.logger.add_scalar("val/acc", c.acc, global_step=self.global_step)
        self.logger.add_scalar("val/miou", c.miou, global_step=self.global_step)
        self.logger.flush()

        return c.miou

    @torch.no_grad()
    def sample(self, epoch=None, sample_dir=None, result_folder=None):
        if sample_dir is None:
            sample_dir = self.cfg.sample_dir
        if sample_dir is None or not os.path.exists(sample_dir):
            return

        if result_folder is None:
            result_folder = os.path.join(self.output_dir, "sample", f"{epoch:04}")
        os.makedirs(result_folder, exist_ok=True)

        self.model.eval()
        files = os.listdir(sample_dir)
        for step, file_name in enumerate(files):
            image = Image.open(os.path.join(sample_dir, file_name)).resize(self.cfg.image_size)
            img = self.cfg.test_transform(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.model(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=os.path.join(result_folder, file_name))

            print(f"[Sample] {step + 1}/{len(files)}", end="\r", flush=True)
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [Sample] {os.path.realpath(result_folder)}")

    def adjust_lr(self, value):
        self.optimizer.adjust_lr(value)

    def adjust_lr_manual(self):
        lr_file = os.path.join(self.cfg.output_dir, "lr.txt")
        if not os.path.exists(lr_file):
            return

        line = open(lr_file, "r").read().strip()
        if line.startswith("#"):
            return

        self.adjust_lr(float(line))
        open(lr_file, "w").write(f"#{line}")

    def save_checkpoint(self, epoch, miou=None, is_best=False):
        state = {
            "epoch": epoch,
            "is_best": is_best,
            "miou": miou,
            "best_miou": self.best_miou,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "loss": self.loss.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(state, os.path.join(self.output_dir, "latest.pth"))
        if is_best:
            self.best_miou = miou
            torch.save(state, os.path.join(self.output_dir, f"{miou:.4f}_{epoch:04}_model.pth"))

    def load_checkpoint(self):
        file = os.path.join(self.output_dir, "latest.pth")
        if not os.path.exists(file):
            return

        state = torch.load(file)
        self.start_epoch = state["epoch"] + 1
        self.model.load_state_dict(state["model"])
        self.loss.load_state_dict(state["loss"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best_miou = state["best_miou"]
        self.global_step = state["global_step"]

    def log(self, msg, end='\n', to_file=True):
        print(msg, end=end, flush=True)
        if to_file:
            print(msg, end='\n', flush=True, file=open(self.log_file, "a+"))
