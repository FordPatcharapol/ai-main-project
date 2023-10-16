import getopt
import sys

try:
    opts, args = getopt.getopt(sys.argv[1:], "hc:v", ["help", "config_path="])
except getopt.GetoptError as err:
    # print help information and exit:
    print(err)  # will print something like "option -a not recognized"
    sys.exit(2)

params = {}

for o, a in opts:
    if o in ("-h", "--help"):
        sys.exit()
    elif o in ("-c", "--config_path"):
        params["config_path"] = a
    elif o in ("-a", "--api_config_path"):
        params["api_config_path"] = a


def get_params(index):
    if index in params:
        return params[index]
