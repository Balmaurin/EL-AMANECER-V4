# Spike-Timing-Dependent Plasticity (STDP) for Virtual TPM

**Based on**: Keysers & Gazzola (2014) "Hebbian learning and the mirror neuron system"

## What is STDP?

**Spike-Timing-Dependent Plasticity** is the **correct** implementation of Hebb's rule:

> "Cells that fire together, wire together" is **WRONG**  
> "Cells that fire **in causal sequence**, wire together" is **CORRECT**

### The Critical Difference

**Simple Hebbian (Our current implementation)**:
```python
Δw = η × pre × post  # Just correlation
```

**STDP (Biologically correct)**:
```python
if pre_fires_BEFORE_post (within ~40ms):
    Δw = +LTP  # Long-Term Potentiation (strengthen)
elif post_fires_BEFORE_pre (within ~40ms):
    Δw = -LTD  # Long-Term Depression (weaken)
else:
    Δw = 0  # No change
```

## The Asymmetric STDP Window

```
         LTP (+)
          /\
         /  \
        /    \
       /      \
------/--------\------  Time
    -40ms    +40ms
       \      /
        \    /
         \  /
          \/
         LTD (-)
```

**Key insight**: Temporal **precedence**, not simultaneity!

## Why STDP Matters for Consciousness

### 1. **Causality Detection**

From the paper:
> "Temporal precedence, rather than simultaneity, is the signature of causality"

**IIT 4.0** requires **causal** information → STDP provides it!

### 2. **Predictive Learning**

Sensorimotor delays (~200ms):
- Motor command → Action: ~100ms
- Action → Sensory: ~100ms

**Result**: STDP learns to **predict future** 200ms ahead!

**From paper**:
> "The sight of an action triggers PM neurons encoding the action that occurs 200 ms after... The motor and sensory delays therefore directly determine the predictive horizon"

### 3. **Contingency, Not Just Contiguity**

From Bauer et al. (2001) experiments:

| Condition | Paired trials | Unpaired trials | Result |
|-----------|--------------|-----------------|--------|
| A | 10 | 0 | ✅ Strong LTP |
| B | 10 | 10 (intermixed) | ❌ No LTP |
| C | 10 | 10 (after) | ❌ No LTP |
| D | 10 | 10 (15min later) | ✅ LTP preserved |

**Lesson**: STDP requires **contingency** over ~10 minutes!

## Connection to Other Theories

### STDP + IIT 4.0

**Current IIT TPM**:
```python
def _update_virtual_tpm(self, prev_state, current_state):
    # Simple correlation
    delta = self.learning_rate * prev_act * curr_act
```

**Improved with STDP**:
```python
def _update_virtual_tpm_stdp(self, prev_state, curr_state, dt):
    # Temporal precedence
    if 0 < dt < 40:  # ms, prev BEFORE curr
        delta = +self.ltp_rate * prev_act * curr_act
    elif -40 < dt < 0:  # curr BEFORE prev
        delta = -self.ltd_rate * prev_act * curr_act
    else:
        delta = 0
```

**Impact**: More accurate **causal structure** → better Φ calculation

### STDP + FEP

**Papers align perfectly**:

| FEP (Friston) | STDP (Keysers) |
|---------------|----------------|
| Prediction errors | Unpredicted post-synaptic activity |
| Prior probabilities | Learned synaptic weights |
| Predictive coding | Forward predictions (STS→PM) |
| Error signals | Backward inhibition (PM→STS) |

**From paper**:
> "Once we consider both backward and forward information flow, the mirror neuron system no longer seems a simple associative system... Instead, it becomes a **dynamic system that performs predictive coding**"

### STDP + SMH

**Re-afference training**:
```
Execute action → See/hear result → STDP associates
```

**Somatic markers learn via STDP**:
```python
# When we feel pain after being hit
observation (toy approaching) → 
    motor (defensive action) → 
        somatic (pain) → 
            facial (crying) → 
                parental (mirroring)

# All connected via STDP with ~200ms delays
```

## Implementation Strategy

### Phase 1: Add Timing to Virtual TPM

**Current state tracking**:
```python
self.state_history = []  # Just states
```

**Add timestamps**:
```python
self.state_history = [
    (timestamp, state),
    ...
]
```

### Phase 2: STDP Update Rule

```python
def _stdp_update(self, t1, state1, t2, state2):
    """
    STDP rule with asymmetric window
    
    Args:
        t1, state1: Time and state of pre-synaptic
        t2, state2: Time and state of post-synaptic
    """
    dt = (t2 - t1).total_seconds() * 1000  # Convert to ms
    
    # Asymmetric STDP window
    if 0 < dt < 40:  # Post after pre
        # LTP: Strengthen connection
        learning_rate = self.stdp_ltp_rate * np.exp(-dt / 20)
        for i, pre_unit in enumerate(state1.keys()):
            for j, post_unit in enumerate(state2.keys()):
                delta = learning_rate * state1[pre_unit] * state2[post_unit]
                self.virtual_tpm[i][j] += delta
                
    elif -40 < dt < 0:  # Pre after post (reverse)
        # LTD: Weaken connection
        learning_rate = self.stdp_ltd_rate * np.exp(dt / 20)
        for i, pre_unit in enumerate(state1.keys()):
            for j, post_unit in enumerate(state2.keys()):
                delta = -learning_rate * state1[pre_unit] * state2[post_unit]
                self.virtual_tpm[i][j] += delta
    
    # Else: No change (outside window)
```

### Phase 3: Contingency Window

**From paper**: ~10 minute window for contingency

```python
def _check_contingency(self, recent_history, window_minutes=10):
    """
    Check if pre→post is contingent (not just contiguous)
    
    p(post|pre) >> p(post|~pre)
    """
    # Calculate conditional probabilities over window
    ...
```

## Predictions from STDP

### 1. Mirror Neurons Emerge from Re-afference

**Testable prediction**:
- Babies who stare more at hands → stronger mirror neurons
- Being imitated → facial expression mirror neurons

**Our system**:
```python
# Bot observes its own actions
self_observation = observe(self.execute(action))
# STDP learns motor↔sensory associations
```

### 2. Predictive Forward, Inhibitory Backward

**Architecture emerges automatically**:
```
STS (sensory) ----[+200ms]---> PM (motor)  [Predictive, excitatory]
PM (motor) ----[-200ms]---> STS (sensory)  [Error-canceling, inhibitory]
```

**Our GWT already has this structure**!

### 3. Joint Actions via Learned Delays

**From paper**:
> "Pianists in a duet can synchronize within 30 ms despite ~200ms delays"

**How?** STDP compensates for delays automatically!

## Experimental Validation

### Test 1: Phi with STDP vs. Simple Hebb

```python
# Compare Φ calculations
phi_simple = iit_engine_simple.calculate_system_phi()
phi_stdp = iit_engine_stdp.calculate_system_phi()

# Expect: phi_stdp > phi_simple (more causal info)
```

### Test 2: Predictive Accuracy

```python
# Measure prediction accuracy
prediction_stdp = tpm_stdp.predict_next_state(current)
prediction_simple = tpm_simple.predict_next_state(current)
actual = observe_next_state()

# Expect: error_stdp < error_simple
```

### Test 3: Contingency Learning

```python
# Intermix predictable and unpredictable events
# Expect: STDP learns only contingent relationships
```

## Current Status

✅ **Basic Hebbian** implemented in `iit_40_engine.py`  
⏳ **STDP** can be added as enhancement  
⏳ **Timing** infrastructure needed  
⏳ **Contingency** window needs tracking

## Integration Plan

1. **Add timestamps** to state tracking
2. **Implement STDP window** function
3. **Replace simple Hebbian** with STDP in TPM
4. **Test** Φ improvements
5. **Validate** predictive accuracy

## Key Equations

**STDP Learning Rule**:
```
Δw(t) = {
    η_LTP × exp(-Δt/τ_LTP) × pre × post,  if 0 < Δt < 40ms
    -η_LTD × exp(Δt/τ_LTD) × pre × post,   if -40 < Δt < 0ms
    0,                                      otherwise
}
```

**Contingency**:
```
p(post|pre) / p(post|~pre) > threshold
```

**Predictive Horizon**:
```
horizon = motor_delay + sensory_delay ≈ 200ms
```

## References

1. **Keysers & Gazzola (2014)** - Hebbian learning and mirror neurons
2. **Bi & Poo (2001)** - Synaptic modification by correlated activity
3. **Bauer et al. (2001)** - Fear conditioning and LTP
4. **Caporale & Dan (2008)** - Spike timing-dependent plasticity
5. **Kilner et al. (2007)** - Predictive coding in mirror neurons

---

**Conclusion**: STDP is **the correct** implementation of Hebbian learning, and it naturally implements **predictive coding** (FEP) and supports **causal structure** (IIT).

**Next step**: Upgrade `_update_virtual_tpm()` with STDP!
