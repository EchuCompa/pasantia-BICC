import sys, os
# This is to import the modules correctly from the parent directory
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)

#There must be a better way to do this, but this is good for now

networkSamplesPath = "networksExamples"