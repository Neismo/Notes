
$$ \begin{align}

\frac{\partial}{\partial \theta}\ln p(y,\theta) &= \frac{\partial}{\partial \theta} (y+a-1)\ln \theta+(N-y+\beta_{0}-1)\ln(1-\theta)+K \\
&= \frac{\partial}{\partial \theta}(y+a-1)\ln \theta + \frac{\partial}{\partial \theta}(N-y+\beta_{0}-1)\ln(1-\theta) + \frac{\partial}{\partial \theta}K \\
&= (y+a-1)\theta^{-1} + (N-y+\beta_{0}-1)(1-\theta)^{-1} \\
& \\
\frac{\partial^2}{\partial \theta^2}\ln p(y,\theta) &= \frac{\partial^2}{\partial^2 \theta} (y+a-1)\ln \theta+(N-y+\beta_{0}-1)\ln(1-\theta)+K \\
&= \frac{\partial^2}{\partial^2 \theta}(y+a-1)\ln \theta + \frac{\partial^2}{\partial^2 \theta}(N-y+\beta_{0}-1)\ln(1-\theta) + \frac{\partial^2}{\partial^2 \theta}K \\
&= -(y+a-1)\theta^{-2} - (N-y+\beta_{0}-1)(1-\theta)^{-2}

\end{align}$$





$$ \begin{align} 
\mathbf{y}=(y^*,\mathbf{y})\mathbf{\mu}=\begin{pmatrix}
0 \\
\mathbf{0}
\end{pmatrix}, \mathbf{\Sigma} = \begin{pmatrix}
4 & K_{X,x^*} \\
K_{X,x^*}^\top & K_{X,X}
\end{pmatrix}
\end{align} $$

$$ \begin{bmatrix}
0.39 & 0.14 \\
0.14 & 0.39
\end{bmatrix} $$$$ \mathbb{E}[\mathbf{f}_{2}] = \mathbb{E}[\mathbf{Af}_{1} + \mathbf{c}] = \mathbf{A}\mathbb{E}[\mathbf{f}_{1}] + \mathbb{E}[\mathbf{c}] = \mathbf{A}\mathbf{0} + \mathbf{0} = \mathbf{0} $$

$$ \mathbb{V}[\mathbf{f}_{2}] = \mathbb{V}[\mathbf{Af}_{1} + \mathbf{c}] = \mathbf{A}^\top\mathbb{V}[\mathbf{f}_{1}]\mathbf{A} + \mathbb{V}[\mathbf{c}] = \mathbf{A}^\top\mathbf{K}\mathbf{A} + \mathbf{I} $$

$$ \mathcal{N}(\mathbf{0}, \mathbf{A^\top KA}+\mathbf{I}) $$


$$ \mathcal{N}\left( \begin{pmatrix}
0.44 \\
1.19 \\
-0.02 \\
-0.10
\end{pmatrix}, \begin{pmatrix}
0.18 & -0.01 & -0.06 & 0.05 \\
-0.01 & 0.25 & 0.01 & -0.01 \\
-0.06 & 0.01 & 0.36 & -0.09 \\
0.05 & -0.01 & -0.09 & 0.71
\end{pmatrix} \right) $$