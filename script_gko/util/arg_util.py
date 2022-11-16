import argparse

MODEL = {
    "mlp"   : 0,
    "xgbc"  : 0,
    "xgbrf" : 0,
    "ddd"   : 0,
    "aaa"   : 0,
}

# get arguments
# TODO: finish setup
def get_args():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp",
                        help="model to train")
    parser.add_argument("--gpu", type=str, default="2",
                        help="gpu to use")
    parser.add_argument("--mode", type=str, default="two",
                        help="mode ")
    arg = parser.parse_args()

    # set gpu number
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu

    return e