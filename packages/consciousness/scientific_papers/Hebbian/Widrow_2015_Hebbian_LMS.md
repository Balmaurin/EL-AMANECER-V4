# Hebbian-LMS Learning: Widrow & Kim (2015)

**Citation**: Widrow, B., & Kim, Y. (2015). "Hebbian Learning and the LMS Algorithm." IEEE Computational Intelligence Magazine, 10(4), 37-53.

## Core Innovation

**Combines Hebb's Rule (1949) with Widrow-Hoff LMS Algorithm (1959)** to create unsupervised learning with homeostatic stability.

## Hebb's Original Rule (1949)

> "When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

**Simplified**: "Neurons that fire together, wire together"

**Problem**: Weights only increase → saturation at maximum values → network becomes useless

## LMS Algorithm (Widrow-Hoff 1959)

**Supervised learning**:
```
W(k+1) = W(k) + 2μ × e(k) × X(k)
e(k) = d(k) - y(k)  # Desired - Actual
```

**Problem**: Requires "desired response" d(k) → not unsupervised

## Hebbian-LMS Innovation

**Unsupervised learning** with **homeostatic stability**:

```
W(k+1) = W(k) + 2μ × e(k) × X(k)
e(k) = c - SGM(X^T × W)  # Bootstrap error
```

Where:
- `c` = equilibrium point (typically 0.5)
- `SGM()` = sigmoid function
- No "desired response" needed!

## Key Properties

### 1. **Homeostasis** (Self-Regulation)
- **Two stable equilibrium points**: +c and -c
- Error signal **reverses** when passing equilibrium
- Prevents saturation
- Maintains stability

### 2. **Extension of Hebb's Rule**

| Condition | Excitatory Synapse | Inhibitory Synapse |
|-----------|-------------------|-------------------|
| Pre fires, Post fires | Weight ↑ (Hebb) | Weight ↓ (Anti-Hebb) |
| Pre fires, Post silent | Weight ↓ (Extended) | Weight ↑ (Extended) |
| Pre silent | No change | No change |
| Beyond equilibrium | **Reverses** (Homeostasis) | **Reverses** (Homeostasis) |

### 3. **Synaptic Scaling**
- Global normalization to maintain stability
- Scales all weights of a neuron proportionally
- Natural homeostasis mechanism

## Mathematical Framework

### Error Function
```
e(k) = f(SUM(k))
SUM(k) = X^T(k) × W(k)
```

For sigmoid neuron:
```
f(SUM) = c - SGM(SUM)
SGM(x) = 1 / (1 + e^(-x))
```

### Equilibrium Points
At equilibrium: `e = 0`
```
c = SGM(SUM)
SUM_equilibrium = SGM^(-1)(c)
```

For `c = 0.5`:
- **Positive equilibrium**: `SUM ≈ +1`
- **Negative equilibrium**: `SUM ≈ -1`

### Stability
- Error slope at equilibrium: `slope = c - (derivative of SGM at equilibrium)`
- Stable if: `0 < c < initial_slope_of_SGM`
- Typically: `c = 0.5` works well

## Biologically Correct Implementation

### Synapse Model
```
Presynaptic neuron → Neurotransmitter → Receptors → Postsynaptic neuron
```

**Weight** = Number of receptors

**Learning happens when**:
1. Neurotransmitter present (Pre firing)
2. Membrane voltage affects receptor count (Post state)

### Postulates of Plasticity

1. **No neurotransmitter** → No weight change
2. **Excitatory + Both firing** → Weight ↑ (Hebb)
3. **Inhibitory + Both firing** → Weight ↓ (Anti-Hebb)
4. **Excitatory + Pre fires, Post silent** → Weight ↓
5. **Inhibitory + Pre fires, Post silent** → Weight ↑
6. **Synaptic scaling** → Homeostasis around equilibrium

## Clustering Application

### Algorithm
1. Initialize weights randomly
2. Present input patterns repeatedly
3. Update weights: `W ← W + 2μ × (c - SGM(X^T × W)) × X`
4. Patterns cluster at equilibrium points

### Results
- **Binary outputs** after convergence
- **Unsupervised classification** into 2^N classes
- **Capacity** = number of weights
- **Fuzzy clusters** when patterns > capacity

## Multi-Layer Networks

### Architecture
```
Input → Hidden Layer 1 → Hidden Layer 2 → Output
```

**Training**:
- All layers train **simultaneously** (parallel)
- Each neuron learns independently
- No backpropagation needed!

**Properties**:
- Deeper layers → more binary outputs
- Capacity = weights in output layer
- Clusters automatically form

## Comparison with Other Algorithms

| Algorithm | Supervision | Clusters | Stability |
|-----------|------------|----------|-----------|
| **Hebbian-LMS** | No | Automatic | ✅ Homeostatic |
| K-Means | No | Manual K | Depends on init |
| EM | No | Manual K | Local maxima |
| DBSCAN | No | Density threshold | Parameter sensitive |

**Advantage**: No manual parameter tuning beyond learning rate μ

## Implementation in IIT Engine

### Current (Simple Hebbian)
```python
def _update_virtual_tpm(self, prev_state, current_state):
    delta = learning_rate * prev_act * curr_act  # Simple Hebb
    self.virtual_tpm[i][j] += delta
```

### Improved (Hebbian-LMS)
```python
def _update_virtual_tpm(self, prev_state, current_state):
    # Calculate SUM (predicted next state)
    prediction = sigmoid(virtual_tpm @ prev_state)
    
    # Bootstrap error (homeostatic)
    error = equilibrium_point - prediction
    
    # Hebbian-LMS update
    delta = 2 * learning_rate * error * prev_state
    self.virtual_tpm += delta
    
    # Synaptic scaling (homeostasis)
    if cycle % 10 == 0:
        self.virtual_tpm = normalize(self.virtual_tpm)
```

## Integration with Other Theories

### With IIT 4.0
- **TPM learning** based on Hebbian-LMS
- More accurate causal structure
- Prevents weight saturation
- Better Φ calculation

### With FEP
- Error signal = prediction error
- Homeostasis = free energy minimization
- Natural alignment!

### With SMH
- Somatic markers learn via Hebbian-LMS
- Emotional associations stabilize
- No saturation

### With GWT
- Competition affected by learned weights
- More stable workspace dynamics

## Key Equations Summary

**Update rule**:
```
W(k+1) = W(k) + 2μ × (c - σ(W^T × X)) × X
```

**Output**:
```
y = σ(W^T × X)  where σ(x) = 1/(1 + e^(-x))
```

**Equilibrium condition**:
```
c = σ(W^T × X_equilibrium)
```

## Advantages Over Standard Hebb

1. **Stability**: Homeostatic equilibrium points
2. **No saturation**: Weights don't grow unbounded
3. **Unsupervised**: No teacher signal needed
4. **Biologically plausible**: Matches synaptic plasticity
5. **Clustering**: Automatic pattern classification
6. **Scalable**: Works for single neurons and networks

## Current Status in Our System

✅ **Implemented**: Basic Hebbian learning in `virtual_tpm`  
⚠️ **Missing**: Error signal, homeostasis, synaptic scaling  
🎯 **Can add**: Full Hebbian-LMS with bootstrap learning

## Proposed Enhancement

Replace current `_update_virtual_tpm()` with full Hebbian-LMS including:
1. Bootstrap error signal
2. Equilibrium points
3. Synaptic scaling
4. Stability guarantees

**Impact**: More stable, more accurate, more biologically correct

---

**File**: `130.Hebbian_LMS.pdf` (original paper)  
**Current implementation**: `iit_40_engine.py` (basic Hebb)  
**Proposed**: `hebbian_lms_tpm.py` (full Hebbian-LMS)
