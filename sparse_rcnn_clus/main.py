from solver.ddp_mix_solver import DDPMixSolver
import os
import warnings
warnings.filterwarnings("ignore")


# from solver.dp_mix_solver import DPMixSolver

# nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 50003 main.py >>train.log 2>&1 &

if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="/home/taoye/code/sparse_rcnnv1-master/config/sparse_rcnn.yaml")
    processor.run()
