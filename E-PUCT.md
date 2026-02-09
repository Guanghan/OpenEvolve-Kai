# E-PUCT Selector Explained

> Evolution-PUCT: Adapting MCTS's PUCT formula for parent program selection in evolutionary algorithms

---

## 1. Core Idea

E-PUCT adapts the PUCT (Predictor + Upper Confidence Bound for Trees) formula from MCTS to evolutionary algorithms:

| MCTS-PUCT Concept | E-PUCT Equivalent | Meaning |
|-------------------|-------------------|---------|
| Q(s,a) Node Value | Fitness(p) | Program's combined_score |
| P(s,a) Prior Probability | Prior(p) | Value converted from LLM's log_prob |
| N(s,a) Visit Count | SelectionCount(p) | Times program was selected as parent |
| N(s) Parent Visits | TotalGenerations | Total evolution generations |
| UCB bonus | ExplorationBonus | Exploration reward |

**Core Insight**: The essence of PUCT is **credit assignment under uncertainty**—balancing exploitation of known good options with exploration of under-tried options. This is exactly what parent selection in evolutionary algorithms needs.

---

## 2. Core Formulas

### 2.1 Selection Score Formula

```
SelectionScore(p) = alpha * Fitness(p) + beta * ExplorationBonus(p) + gamma * Novelty(p)
```

Where:
- **alpha = 0.6**: Fitness weight (exploitation)
- **beta = 0.3**: Exploration weight (exploration)
- **gamma = 0.1**: Novelty weight (diversity)

### 2.2 Exploration Bonus Formula (PUCT-style)

```
ExplorationBonus(p) = c_puct * Prior(p) * sqrt(TotalGenerations) / (1 + SelectionCount(p))
```

Where:
- **c_puct = 1.5**: PUCT constant, controls exploration intensity
- **Prior(p)**: Prior value converted from LLM log_prob
- **TotalGenerations**: Total evolution generations
- **SelectionCount(p)**: Times this program was selected as parent

---

## 3. Formula Visualization

```
+-----------------------------------------------------------------------------+
|                     E-PUCT Selection Score Calculation                       |
+-----------------------------------------------------------------------------+

                           SelectionScore(p)
                                  |
          +-----------------------+-------------------------+
          |                       |                         |
          v                       v                         v
    +-----------+          +-----------+            +-----------+
    | alpha=0.6 |          | beta=0.3  |            | gamma=0.1 |
    |     x     |          |     x     |            |     x     |
    |  Fitness  |          |Exploration|            |  Novelty  |
    +-----+-----+          +-----+-----+            +-----+-----+
          |                      |                        |
          v                      v                        v
    +-----------+          +-------------------+    +-----------+
    |normalized |          |      c_puct       |    |  0.9^count|
    |  fitness  |          |        x          |    |           |
    |   [0,1]   |          |     Prior(p)      |    |unexplored |
    +-----------+          |        x          |    |   = 1.0   |
                           |   sqrt(TotalGen)  |    | explored  |
                           |  ---------------  |    |   = low   |
                           | 1+SelectCount(p)  |    +-----------+
                           +-------------------+
```

---

## 4. Component Details

### 4.1 Fitness Component (Exploit Known Good Programs)

```python
# Normalize fitness to [0, 1]
fitness_range = self._max_fitness - self._min_fitness
if fitness_range > 0:
    normalized_fitness = (fitness - self._min_fitness) / fitness_range
else:
    normalized_fitness = 0.5
```

**Purpose**: Ensure high-fitness programs have higher selection probability.

### 4.2 Exploration Bonus Component (Explore Under-tried Programs)

```python
exploration_bonus = (
    self.config.c_puct           # 1.5
    * prior                       # converted from log_prob
    * math.sqrt(total_generations)
    / (1 + selection_count)
)

# Clamp range
exploration_bonus = max(0.1, min(2.0, exploration_bonus))
```

**Key Insights**:
- Low `selection_count` -> High bonus -> Encourage trying under-explored programs
- Increasing `total_generations` -> Higher bonus -> More exploration needed in later stages
- High `prior` -> High bonus -> Prioritize mutations LLM considers "natural"

### 4.3 Prior Calculation (From log_prob)

```python
def _logprob_to_prior(self, log_prob: float) -> float:
    """
    Convert log probability to [0, 1] range prior

    log_prob is typically in [-10, 0] range
    Uses sigmoid function for mapping
    """
    x = (log_prob + 5) / 2
    return 1 / (1 + math.exp(-x))

# Mapping:
# log_prob = 0   -> prior ~ 0.92 (high confidence, LLM considers natural)
# log_prob = -5  -> prior ~ 0.50 (medium confidence)
# log_prob = -10 -> prior ~ 0.08 (low confidence, LLM considers unnatural)
```

### 4.4 Novelty Component (Encourage Exploring New Regions)

```python
def _compute_novelty(self, feature_coords: Tuple[int, ...]) -> float:
    """
    Compute novelty based on MAP-Elites feature coordinates
    """
    count = self._feature_counts.get(feature_coords, 0)

    if count == 0:
        return 1.0  # Unexplored region -> highest novelty
    else:
        return 0.9 ** count  # Exponential decay

# Examples:
# Region (2,3) visited 0 times -> novelty = 1.0
# Region (1,1) visited 5 times -> novelty = 0.59
# Region (0,0) visited 10 times -> novelty = 0.35
```

---

## 5. Adaptive Exploration Mechanism

E-PUCT dynamically adjusts exploration intensity based on recent evolution progress:

```python
def _get_exploration_multiplier(self) -> float:
    """Adjust exploration intensity based on recent success rate"""

    # Calculate recent success rate
    recent_success_rate = ...

    if recent_success_rate < 0.3:
        return 1.5  # Stagnating -> increase exploration
    elif recent_success_rate > 0.7:
        return 0.7  # Good progress -> reduce exploration, more exploitation
    else:
        return 1.0  # Normal
```

**Logic**:
- Low success rate = current direction may be bad -> increase exploration for new directions
- High success rate = current direction is good -> reduce exploration, focus on exploitation

---

## 6. Concrete Calculation Example

Assume at generation 30 of a Circle Packing task:

### Program A (High fitness, selected many times)

```
+-- fitness = 0.75 -> normalized = 0.8
+-- selection_count = 5 (selected 5 times)
+-- log_prob = -2.0 -> prior = 0.82
+-- feature_coords = (2, 3), visited 3 times -> novelty = 0.73

exploration_bonus = 1.5 x 0.82 x sqrt(30) / (1+5) = 1.12

score = 0.6 x 0.8 + 0.3 x 1.12 + 0.1 x 0.73
      = 0.48 + 0.34 + 0.07
      = 0.89
```

### Program B (Medium fitness, rarely selected)

```
+-- fitness = 0.65 -> normalized = 0.6
+-- selection_count = 1 (selected only once) <- under-explored
+-- log_prob = -4.0 -> prior = 0.62
+-- feature_coords = (4, 1), never visited -> novelty = 1.0 <- new region

exploration_bonus = 1.5 x 0.62 x sqrt(30) / (1+1) = 2.55
                  -> clamped to 2.0 (maximum limit)

score = 0.6 x 0.6 + 0.3 x 2.0 + 0.1 x 1.0
      = 0.36 + 0.60 + 0.10
      = 1.06 <- Higher!
```

### Result

**Program B is selected** (despite lower fitness, higher exploration bonus)

This is the core value of PUCT: giving under-explored options a fair chance.

---

## 7. Complete Selection Flow

```
+-----------------------------------------------------------------------------+
|                        E-PUCT Parent Selection Flow                          |
+-----------------------------------------------------------------------------+

                    Candidate Parent Pool (from islands)
                    [P1, P2, P3, P4, P5, ...]
                              |
                              v
+-----------------------------------------------------------------------------+
|  For each candidate program Pi, calculate SelectionScore                     |
|                                                                              |
|  1. Get fitness (combined_score) and normalize to [0,1]                      |
|                                                                              |
|  2. Get Prior:                                                               |
|     - Has log_prob -> sigmoid transform                                      |
|     - No log_prob -> use default 0.5                                         |
|                                                                              |
|  3. Calculate ExplorationBonus:                                              |
|     c_puct x prior x sqrt(total_gen) / (1 + selection_count)                 |
|                                                                              |
|  4. Calculate Novelty:                                                       |
|     novelty = 0.9^(feature_region_visit_count)                               |
|                                                                              |
|  5. Adaptive adjustment:                                                     |
|     exploration_bonus x exploration_multiplier                               |
|                                                                              |
|  6. Weighted sum:                                                            |
|     score = 0.6*fitness + 0.3*exploration + 0.1*novelty                      |
|                                                                              |
+-----------------------------------------------------------------------------+
                              |
                              v
                    Sort by score descending
                    [P3(1.06), P1(0.89), P5(0.85), ...]
                              |
                              v
                    Select Top-K as parent programs
                              |
                              v
              Update selection_count += 1 for selected programs
                              |
                              v
                      Return selected parents
```

---

## 8. Comparison with OpenEvolve's Original Sampling

| Dimension | OpenEvolve Original | E-PUCT Enhanced |
|-----------|---------------------|-----------------|
| **Exploitation** | 70% exploitation (select best) | alpha=0.6 fitness weight |
| **Exploration** | 30% exploration (pure random) | beta=0.3 PUCT bonus (guided exploration) |
| **Diversity** | MAP-Elites grid | gamma=0.1 novelty bonus (extra reward) |
| **Memory** | None | Track selection_count |
| **Adaptation** | None | Dynamic adjustment based on success rate |
| **Prior Guidance** | None | Use LLM log_prob as prior |

---

## 9. Data Structures

```python
@dataclass
class SelectionStats:
    """Selection statistics for each program"""
    program_id: str
    selection_count: int = 0           # Times selected as parent
    successful_offspring: int = 0       # Number of successful children
    failed_offspring: int = 0           # Number of failed children
    total_improvement: float = 0.0      # Cumulative improvement
    log_prob_prior: Optional[float]     # Cached prior
    last_selected_generation: int = 0   # Last generation when selected

class EPUCTSelector:
    stats: Dict[str, SelectionStats]    # Stats for each program
    total_generations: int              # Total generations
    total_selections: int               # Total selections made
    _feature_counts: Dict[Tuple, int]   # Feature region visit counts
```

---

## 10. Configuration Parameters

```yaml
epuct_selector:
  enabled: true

  # Weight configuration
  alpha: 0.6      # fitness weight (exploitation)
  beta: 0.3       # exploration weight (exploration)
  gamma: 0.1      # novelty weight (diversity)

  # PUCT parameters
  c_puct: 1.5     # PUCT constant

  # Prior configuration
  use_logprob_prior: true
  default_prior: 0.5

  # Novelty configuration
  use_novelty_bonus: true
  novelty_decay: 0.9

  # Adaptive exploration
  adaptive_exploration: true
  min_exploration_bonus: 0.1
  max_exploration_bonus: 2.0
```

### Configuration Recommendations

```yaml
# More exploration (suitable for early stage or stagnation)
epuct_selector:
  alpha: 0.4
  beta: 0.5
  gamma: 0.1
  c_puct: 2.0

# More exploitation (suitable for late-stage convergence)
epuct_selector:
  alpha: 0.8
  beta: 0.15
  gamma: 0.05
  c_puct: 1.0
```

---

## 11. Key Insights

### Why is E-PUCT Better Than Pure Random Exploration?

| Pure Random Exploration | E-PUCT Exploration |
|-------------------------|-------------------|
| May repeatedly select the same "okay" program | selection_count ensures every program gets a chance |
| May ignore potentially good but untried programs | Untried programs get high exploration bonus |
| Exploration direction has no guidance | Prior guides toward "reasonable" directions per LLM |
| Doesn't consider feature space diversity | Novelty encourages exploring new feature regions |
| Static strategy | Adaptive mechanism responds to evolution stagnation |

### The Essence of PUCT

```
The core of PUCT is:
"Give every option a fair chance, but bias toward those that look promising"

In evolution:
- "Promising" = High fitness + High prior (LLM considers it reasonable)
- "Fair chance" = Unselected programs get extra exploration bonus
```


