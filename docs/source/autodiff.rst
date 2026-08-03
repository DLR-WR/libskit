About Automatic Differentiation
===============================

For now we are using forward mode automatic differentiation.
This is nice and easy to implement, but if we calculate the gradients to all parameters of the LTE model, this will result in the irradiance/spectrum being calculated N times (where N is the number of parameters of the LTE model).

There is a way around this, which is to use reverse mode autodiff.
I have not tried this, but my gut feeling is that the benefit of only needing to execute the function once with reverse mode autodiff is offset by the more expensive calculation of an expression tree.

Eitherway, this scales linearly in the number of parameters of the plasma model, so usually it will be on the order of 10s.
