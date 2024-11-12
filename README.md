# Approximate Squaring Visualization

![Approximate Squaring](./approximate_squaring.gif)

This is a 3D visualization of the "approximate squaring" function studied in the paper "Approximate Squaring" by J. C. Lagarias and N. J. A. Sloane (2004). 

## Background

The approximate squaring map f(x) is defined as:

```
f(x) = ⌈x⌉x
```

where ⌈x⌉ denotes the ceiling function. When starting with a rational number r = l/d where l is the numerator and d is the denominator, the paper studies how many iterations it takes before reaching an integer value.

## Visualization Features

The program creates a 3D animated surface plot showing:
- X-axis: Numerator values (1 to 10,000)
- Y-axis: Denominator values (2 to 20)
- Z-axis: Number of steps required to reach an integer

## References

Lagarias, J. C., & Sloane, N. J. A. (2004). Approximate Squaring. Experimental Mathematics, 13(1), 113-128.

## License

MIT License