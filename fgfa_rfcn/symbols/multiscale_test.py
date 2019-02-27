import mxnet as mx
input_array = mx.nd.array([[[[1,2], [3,4]], [[5,6], [7,8]]], [[[3,2], [5,3]], [[6,1], [7,8]]]])

a = mx.symbol.Variable('a')
b = mx.symbol.sum(data=a, axis=1)
c = mx.symbol.transpose(data = b, axes=(1,0,2))
d = mx.symbol.SoftmaxActivation(data = c, mode='channel')
e = mx.symbol.transpose(data = d, axes=(1,0,2))



f = e.bind(mx.cpu(), {'a': input_array})
y = f.forward()
print y[0].asnumpy()
