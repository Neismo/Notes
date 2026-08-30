
## Warps:

As thread IDs are done as 
- 2D block (Dx, Dy): $ID = x + y * Dx$ <-- COLUMN WISE, but data is ROW WISE 
	- in practice: `j, i = cuda.grid(2)`!
- 3D block (Dx): $ID = x + y*Dx + z*Dy*Dx$ <-- INEFFICIENT W.R.T DATA STORAGE
	- In practice: `z, y, x = cuda.grid(3)`

## Parallel Fraction & SpeedUps
p = parallel fraction
n = number of cores
S = speedUp
T = total time

$$S_n = \frac{1}{1 - p + p/n}$$
Speed on 1 core compared to $n$:
$$ T_1 = S_n\cdot T_n $$
	$$ p = \frac{n(S-1)}{S(n-1)} = \frac{nS -n}{nS - S}$$

$$ T_\text{serial} + \frac{T_\text{parrallel}}{n} = T_n $$

## Broadcasting
Shape 1: $N \times H \times 1$
Shape 2 $H \times W$

Align Shapes with LEFT PADDING 1:
$$ \begin{align}
N\times H \times 1 \\
1\times H \times W
\end{align} $$
Then match 1's with the other matrices (ANYTHING OTHER THAN 1s WILL FAIL BROADCASTING)
$$  \begin{align}
N\times H \times W \\
N\times H \times W
\end{align}  $$
Now it works!

### None indexing (add dimensions)
`matrix[None]` adds a DIMENSION. If not specified, it adds LEFT.
- 2D Matrix X:
	- `X[None].shape = (1, W, H)`
	- `X[:, None].shape = (W, 1, H)`
	- `X[:, :, None].shape = (W, H, 1)`
So as you can see, we need to select the dimension so we insert _after_ those!

## Profiling NSYS

CUDA API SUMMARY:
![[Pasted image 20260528165514.png]]
- **Time (%)**: not total % of runtime, but of Cuda API calls / this section
- **Total time (ns)**: same but in nano-seconds
- **Num Calls** number of calls

GPU Kernel Summary
![[Pasted image 20260528165701.png]]
- **Name:** Can see kernel information (above is a single doubling array kernel)
- **Instances**: 
- **Grid XYZ**: Number of blocks (977) - above was 1D
- **BlockXYZ**: Theads per block (1024) - above was 1D

GPU MemOps Summary (time and size)
![[Pasted image 20260528165936.png]]
- **Operation**: HtoD (CPU to GPU) / DtoH (GPU to CPU)
- **Count**: how many times the operation was done (above was 2 times for each way; one for the input and one for output). If allocation of output was done on the device, we would only have 1 transfer each way!


# Sizes
UINT32, FLOAT32 = 4 bytes
INT64, FLOAT64    = 8 bytes

1 MB = 1.000 KB = 1.000.000 bytes

| Bits   | Signed Integer Range            | Unsigned Integer Range |
| :----- | :------------------------------ | :--------------------- |
| **8**  | -128 to 127                     | 0 to 255               |
| **16** | -32,768 to 32,767               | 0 to 65,535            |
| **32** | -2,147,483,648 to 2,147,483,647 | 0 to 4,294,967,295     |
| **64** | $-2^{63}$ to $2^{63}-1$         | 0 to $2^{64}-1$        |

# Zarr
If data is in $X\times Y$ shape, and you need to sum over rows (i.e X), then you want a "process" function that works in chunks of (1, Y)!

# Memory

## NUMPY: 
- Numpy operations will release the GIL (good for threading)
- MEMMAP: does NOT take RAM when; only those we read/write takes RAM!