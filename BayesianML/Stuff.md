
$$ \begin{align}

\frac{\partial}{\partial \theta}\ln p(y,\theta) &= \frac{\partial}{\partial \theta} (y+a-1)\ln \theta+(N-y+\beta_{0}-1)\ln(1-\theta)+K \\
&= \frac{\partial}{\partial \theta}(y+a-1)\ln \theta + \frac{\partial}{\partial \theta}(N-y+\beta_{0}-1)\ln(1-\theta) + \frac{\partial}{\partial \theta}K \\
&= (y+a-1)\theta^{-1} + (N-y+\beta_{0}-1)(1-\theta)^{-1} \\
& \\
\frac{\partial^2}{\partial \theta^2}\ln p(y,\theta) &= \frac{\partial^2}{\partial^2 \theta} (y+a-1)\ln \theta+(N-y+\beta_{0}-1)\ln(1-\theta)+K \\
&= \frac{\partial^2}{\partial^2 \theta}(y+a-1)\ln \theta + \frac{\partial^2}{\partial^2 \theta}(N-y+\beta_{0}-1)\ln(1-\theta) + \frac{\partial^2}{\partial^2 \theta}K \\
&= -(y+a-1)\theta^{-2} - (N-y+\beta_{0}-1)(1-\theta)^{-2}

\end{align}$$