x <- seq(-3, 3, length=1000000)
y <- dnorm(x, mean=0, sd=1)
plot(x, y, type="l", lwd=1, ylab="g(x)")