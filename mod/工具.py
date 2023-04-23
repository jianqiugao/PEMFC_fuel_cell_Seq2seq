def normal_data(x,max_,min_):
    x = min(max(x,min_),max_)
    x = (x -min_)/(max_-min_)
    return x
def unormal_data(x,max_,min_):
    x = x*(max_-min_)+min_
    return x


