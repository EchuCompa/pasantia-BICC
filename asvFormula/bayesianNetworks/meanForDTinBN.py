import sys, os
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path + "/..")

#Horrible code, will fix later