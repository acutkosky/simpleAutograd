# simpleAutograd

This is a (relatively) simple automatic differentation package. It implements backprop/reverse-mode differentiation. To use, you must wrap numpy arrays in `Variable` objects. Then, declare and apply `Operation` objects to these variables. The output will be a new `Variable`. By calling `.backward()`, each of the input `Variable` objects will have their `.grad` fields updated to the gradient. Note that `.backward()` is best called on scalar variables, unless you really know what you are doing.
