from torch import empty
from torch.nn.functional import one_hot

class OneHot(object):
    """ 
    Takes LongTensor with index values of shape (*) and returns a tensor of shape
    (*, num_classes) that have zeros everywhere except where the index of last
    dimension matches the corresponding value of the input tensor, in which case
    it will be 1.
    """
    def __init__(self, column=0, num_classes=-1):
        self.column = column
        self.num_classes = num_classes

    def __call__(self, graph):
        enc = one_hot(
            graph.x[:, self.column],
            num_classes=self.num_classes
        )

        x = empty(
            (enc.shape[0], enc.shape[1] + graph.x.shape[1] - 1)
        )
    
        x[:, :self.column] = graph.x[:, :self.column]
        x[:, self.column:(self.column + enc.shape[1])] = enc
        x[:, (self.column + enc.shape[1]):] = graph.x[:, (self.column + 1):]
        graph.x = x

        return graph

    def __repr__(self):
        return f"OneHot(self.column={self.column}, num_classes={self.num_classes})"
