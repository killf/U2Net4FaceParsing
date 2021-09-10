from solver import Solver
from config import Config

if __name__ == '__main__':
    cfg = Config()
    cfg.data_dir = "/home/renpeng/dataset/CelebAMask-HQ_processed2"
    cfg.sample_dir = "/home/renpeng/dataset/testset_210720_aligned"
    cfg.batch_size = 16
    cfg.epochs = 200

    solver = Solver(cfg)
    solver.train()
