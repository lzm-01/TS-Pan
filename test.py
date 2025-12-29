from utils.config  import get_config
from solver.testsolver_1 import Testsolver

if __name__ == '__main__':
    cfg = get_config('option.yml')
    solver = Testsolver(cfg)
    solver.run()
