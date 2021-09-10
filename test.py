from solver import Solver
from config import Config

if __name__ == '__main__':
    cfg = Config()
    cfg.data_dir = "/data/face/parsing/dataset/ibugmask_release"
    cfg.model_args.backbone = "STDCNet1446"
    cfg.model_args.pretrain_model = "snapshot/STDCNet1446_76.47.tar"

    solver = Solver(cfg)
    solver.sample(sample_dir="/data/face/parsing/dataset/testset_210720_aligned", result_folder="result")
