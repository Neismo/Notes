Event by event; sub-sampling
- Continuous but asynchronous time (jump between events)
- Described by *state variables*; (waiting, check-out etc.)

Core elements:
- Real-time clock
- State variables
- Event list(s)
- Statistical Accumulators

**Event-by-event**: (loop)
- Advance simulation clock
- Invoke event-corresponding routine
- Collect statistics
- Update system state variables
- Generate + schedule future events in event list
Event -> update state -> event ...

*Burn-in period*: cold-start, biased initial states, length determined experimentally
*Estimates*: confidence intervals using sub-samples

### Queueing systems
Characteristics:
- Arrival process (Poisson) $A(t)$
- Service time distribution(s) (Exponential) $S(t)$
	- Finite or infinite waiting room capacity ($K$)
- Service unit(s) {1 or more - $N$}
- Priorities
- Queueing discipline

Commonly described with Kendall:
- $A(t)$: M (exponential) or G (general)
- $S(t)$: M or G 
- $N$: number of servers
- $K$: capacity of system (sometimes waiting room capacity only)

Arrival process usually modeled with *Poisson*:
- Independent and exponentially distributed waiting times $s_i\sim\exp(\lambda)$.
- Number of arrivals in disjoint times are independent
- Number of arrivals up to time $t$ is Poisson distributed $N(t) = \text{Poisson}(t\lambda)$

**Performance measures**:
- Mean waiting time
- Variance of waiting time
- Quantiles

- Blocking probabilities
- Utilization of servers
- queue-length distribution

### Sub-sampling
Do batches (sub-samples) and compute estimates from those. Variance between batches approximate the overall variance.

The standard deviation of the estimator decreases proportionally to $1/\sqrt{n}$.

Get $n$ estimates $\hat{\theta}_i$;
	$$ \bar{\theta} = \frac{1}{n}\sum_i^n\hat{\theta}_i $$$$ sd_\theta^2 = \frac{1}{n-1}\left(\sum_i^n\hat{\theta}_i^2 - n\bar{\theta}^2\right) $$
Ferry example (Exercise?)
- States:
	- C = [2] channels (all un-occupied at start)
	- B = [4,2] berths (4 in each location, all occupied except 1 at each)
	- F_L = [6] ferry loaded (not at start)
	- F_S = [6] ferry ready to sail (not at start)
- Events:
	- Arrive/Depart harbour
	- Arrive/Depart berth
	- Loading complete
	- Ferry ready to sail
	- Ferry arrive/depart channel
Examples:
- t: arrive_harbour_1_boat_3
	- if (C[1] = 0 ) schedule(arrive_channel, t=t+$\epsilon$), C[1] = 1, FP[3] = C1
	- else schedule(arrive_channel, t=t+T[leave_channel])
- t=t+$\epsilon$ arrive_channel_1_boat_3
	- if (B[1,:] =0) schedule(arrive_berth_14, t=t+$\Delta$), C[1] = 0, FP[3] = B14, B[1, 4]=1, FL[3]=0, FS[3]=0
- t=t+$\epsilon+\Delta$ arrive_berth_14_3
	- schedule(loaded, t=t+exp($\lambda$)), FL[3]=1

## Exercise 4:
B is for blocking probability; P(m) of being in state m, with m being capacity. P(m) is then probability of being in state of full capacity.