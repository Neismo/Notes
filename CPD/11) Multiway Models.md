
**Tensors**: $N$-dimensional array for representing data.
- Allows *unique* decompositions
- Identifying components when facing poor signal-to-noise ratio (SNR)
- Handle data with (large amounts) of missing data
- Explicitly multi-way structure
- *Increased* interpretability

**Tensor** are denoted with caligraphics:
$$ \mathcal{X}\in \mathbb{R}^{\mathcal{T}_{1},\dots,\mathcal{T}_{N}}, $$
where $N$ is the order; $\mathcal{X}$ is the measured response. **Frobenius Norm**:
$$ ||\mathcal{X}_{F}|| = \sqrt{ \mathcal{} } $$

## Tucker
![[Pasted image 20260430082518.png]]

Considering 3-way tensors (order 3):
$\mathcal{G}$ is the core tensor, used with loading matrices A, B, and C, to define the tensor $\mathcal{X}$:
$$ \mathcal{X} = \mathcal{G}\times_{1}A\times_{2}B\times_{3}C $$

**Matrix Form**:
![[Pasted image 20260430083047.png]]

**Tucker Optimization**: do least squares in alternating matter until convergence:
![[Pasted image 20260430083100.png]]



## PARAFAC

A special case of Tucker, where the core tensor $\mathcal{G}$ is super-diagonal (1s along diagonal, 0 otherwise). 