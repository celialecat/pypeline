from classy_sz import Class as Class_sz
# Initialize classy_sz
classy_sz = Class_sz()
# classy_sz.set(params)
classy_sz.set({'pressure_profile':'GNFW'})  # Set parameters
classy_sz.compute_class_szfast()  # Required before calling other functions