# -*- coding: utf-8 -*-
"""
ORBITOFRONTAL_CORTEX - EXTENDED ENTERPRISE
=========================================

Implementación enterprise del OFC:
- Estimaciones de valor con histórico y persistencia
- Aprendizaje por error de predicción (PE) con learning rate adaptativo
- Detección y manejo de reversals (change-point)
- Políticas de decisión: greedy, epsilon-greedy, softmax (Boltzmann)
- Integración multi-atributo y compatibilidad con vmPFC/RAG
- Hooks / callbacks para orquestador MCP (on_update, on_decision)
- Persistencia SQLite (opcional)
- Métricas y export_state

Autor: Adaptado para Sheily-AI / Kimi
Fecha: 2025-11-25
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import time
import sqlite3
import json
import math
import os
import logging
import numpy as np

logger = logging.getLogger("ofc_extended")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# ---------------------
# Persistence helper (SQLite)
# ---------------------
class OFCDB:
    def __init__(self, path: str = "ofc_values.db"):
        self.path = path
        init_needed = not os.path.exists(path)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if init_needed:
            self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS value_estimates (
            stimulus_id TEXT PRIMARY KEY,
            expected_value REAL,
            confidence REAL,
            history TEXT,
            last_update_ts REAL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY,
            ts REAL,
            selected TEXT,
            policy TEXT,
            details TEXT
        )
        """)
        self.conn.commit()

    def save_value(self, ve: "ValueEstimate"):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO value_estimates (stimulus_id, expected_value, confidence, history, last_update_ts)
        VALUES (?, ?, ?, ?, ?)
        """, (ve.stimulus_id, float(ve.expected_value), float(ve.confidence), json.dumps(ve.history), float(ve.last_update.timestamp())))
        self.conn.commit()

    def load_value(self, stimulus_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM value_estimates WHERE stimulus_id = ?", (stimulus_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def save_decision(self, decision_id: str, selected: str, policy: str, details: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO decisions (id, ts, selected, policy, details) VALUES (?, ?, ?, ?, ?)",
                    (decision_id, time.time(), selected, policy, json.dumps(details)))
        self.conn.commit()

    def recent_decisions(self, limit: int = 20):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# ---------------------
# ValueEstimate (robust)
# ---------------------
@dataclass
class ValueEstimate:
    stimulus_id: str
    expected_value: float = 0.0
    confidence: float = 0.1
    history: List[float] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.utcnow)

    # internal tracking for reversal detection
    pe_history: List[float] = field(default_factory=list)  # prediction errors

    def update(self, outcome: float, learning_rate: float):
        """
        Update rule with prediction error and adaptive confidence.
        Returns prediction_error.
        """
        prediction_error = float(outcome - self.expected_value)
        # Update expected value (simple delta rule)
        self.expected_value += learning_rate * prediction_error
        # Append history (bounded)
        self.history.append(float(outcome))
        if len(self.history) > 500:
            self.history.pop(0)
        # track PE
        self.pe_history.append(abs(prediction_error))
        if len(self.pe_history) > 500:
            self.pe_history.pop(0)
        # Update confidence: increase with consistent small PE, decrease after big surprises
        recent_pe_mean = float(np.mean(self.pe_history[-20:])) if self.pe_history else 0.0
        # heuristic: confidence inversely related to recent PE
        self.confidence = float(np.clip(1.0 - (recent_pe_mean / (abs(self.expected_value) + 1.0)), 0.01, 0.999))
        self.last_update = datetime.utcnow()
        return float(prediction_error)


# ---------------------
# Orbitofrontal Cortex Extended
# ---------------------
class OrbitofrontalCortex:
    def __init__(self,
                 system_id: str,
                 persist: bool = True,
                 db_path: str = "ofc_values.db",
                 base_learning_rate: float = 0.3,
                 discount_factor: float = 0.95,
                 reversal_pe_threshold: float = 0.6,
                 reversal_window: int = 10,
                 logging: bool = True):
        self.system_id = system_id
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.values: Dict[str, ValueEstimate] = {}
        self.base_lr = float(base_learning_rate)
        self.discount_factor = float(discount_factor)
        self.reversal_pe_threshold = float(reversal_pe_threshold)  # PE magnitude to consider reversal
        self.reversal_window = int(max(3, reversal_window))  # how many recent PEs to check
        self.total_evaluations = 0
        self.reversals_detected = 0
        self.decisions_made = 0
        self.db = OFCDB(db_path) if persist else None
        self.on_update: Optional[Callable[[ValueEstimate, Dict[str, Any]], None]] = None
        self.on_decision: Optional[Callable[[Dict[str, Any]], None]] = None
        if logging:
            logger.setLevel(logging.INFO)
        logger.info("OFC initialized id=%s lr=%.3f discount=%.3f", system_id, self.base_lr, self.discount_factor)
        # load persisted values (lazy loading)
        if self.db:
            self._lazy_load_keys = None

    # ---------------------
    # Internal helpers
    # ---------------------
    def _get_value(self, stimulus_id: str) -> ValueEstimate:
        if stimulus_id in self.values:
            return self.values[stimulus_id]
        # try DB
        if self.db:
            row = self.db.load_value(stimulus_id)
            if row:
                ve = ValueEstimate(
                    stimulus_id=stimulus_id,
                    expected_value=float(row["expected_value"]),
                    confidence=float(row["confidence"]),
                    history=json.loads(row["history"]) if row["history"] else [],
                    last_update=datetime.utcfromtimestamp(float(row["last_update_ts"]))
                )
                self.values[stimulus_id] = ve
                return ve
        # create new
        ve = ValueEstimate(stimulus_id=stimulus_id, expected_value=0.0, confidence=0.1)
        self.values[stimulus_id] = ve
        return ve

    def _persist_value(self, ve: ValueEstimate):
        if self.db:
            try:
                self.db.save_value(ve)
            except Exception:
                logger.exception("OFC: failed to persist value")

    def _adaptive_lr(self, ve: ValueEstimate) -> float:
        """
        Adaptive learning rate: higher when confidence low or after a detected reversal.
        """
        # if low confidence -> increase LR
        lr = self.base_lr * (1.0 + (0.8 * (1.0 - ve.confidence)))
        return float(np.clip(lr, 0.01, 0.9))

    def _check_reversal(self, ve: ValueEstimate) -> bool:
        """
        Detect reversal: if recent PEs average > threshold and sustained.
        """
        if len(ve.pe_history) < self.reversal_window:
            return False
        recent = ve.pe_history[-self.reversal_window:]
        mean_pe = float(np.mean(recent))
        if mean_pe >= self.reversal_pe_threshold:
            return True
        return False

    # ---------------------
    # Public API: evaluation & learning
    # ---------------------
    def evaluate_stimulus(self, stimulus: Dict[str, Any]) -> float:
        """
        Return expected value for stimulus (create if missing).
        stimulus must have 'id' key.
        """
        self.total_evaluations += 1
        sid = stimulus.get("id", str(stimulus))
        ve = self._get_value(sid)
        return float(ve.expected_value)

    def update_value(self, stimulus_id: str, outcome: float, learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Update stored value estimate given observed outcome.
        Returns dict with prediction_error and reversal_flag.
        """
        ve = self._get_value(stimulus_id)
        lr = learning_rate if learning_rate is not None else self._adaptive_lr(ve)
        pe = ve.update(float(outcome), lr)
        reversal = False
        if self._check_reversal(ve):
            reversal = True
            self.reversals_detected += 1
            # adapt to reversal: reduce confidence to relearn fast; temporarily increase lr
            ve.confidence = max(0.01, ve.confidence * 0.4)
            # optional: boost learning rate for immediate further updates (handled by adaptive_lr next call)
            logger.info("OFC: reversal detected for %s (pe=%.3f). Confidence reduced.", stimulus_id, pe)
        # persist & callback
        self._persist_value(ve)
        if self.on_update:
            try:
                self.on_update(ve, {"stimulus_id": stimulus_id, "pe": pe, "reversal": reversal})
            except Exception:
                logger.exception("OFC on_update hook error")
        return {"prediction_error": float(pe), "reversal": reversal, "expected_value": float(ve.expected_value), "confidence": float(ve.confidence)}

    # ---------------------
    # Multi-attribute integration
    # ---------------------
    def integrate_attributes(self, stimulus: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> float:
        """
        Integrate multiple attributes into a single scalar value.
        Default weights: monetary 0.4, social 0.3, emotional 0.2, novelty 0.1
        """
        if weights is None:
            weights = {"monetary": 0.4, "social": 0.3, "emotional": 0.2, "novelty": 0.1}
        total = 0.0
        for attr, w in weights.items():
            v = stimulus.get(attr)
            if v is None:
                continue
            try:
                val = float(v)
            except Exception:
                continue
            total += w * val
        return float(total)

    # ---------------------
    # Decision policies
    # ---------------------
    def choose_action(self,
                      options: List[Dict[str, Any]],
                      policy: str = "epsilon_greedy",
                      epsilon: float = 0.05,
                      softmax_temp: float = 1.0,
                      integrate_attrs: bool = False,
                      attr_weights: Optional[Dict[str, float]] = None,
                      persist_decision: bool = True) -> Dict[str, Any]:
        """
        Choose an action among options.
        Each option can be:
            {'id': 'a', 'value': 1.2} or {'id': 'a', 'outcomes':[{'value':v,'prob':p},...]} or multi-attr stimulus
        Policies:
            - greedy: choose max EV deterministically
            - epsilon_greedy: explore with prob eps
            - softmax: sample via Boltzmann with temperature
        Returns dict {selected_option, scores, policy_used, decision_id}
        """
        self.decisions_made += 1
        if not options:
            return {}

        # compute values
        scores = []
        for opt in options:
            # if integrate_attrs -> use integrate_attributes
            if integrate_attrs:
                val = self.integrate_attributes(opt, weights=attr_weights)
            else:
                # use stored expected value if stimulus present
                sid = opt.get("id")
                if sid and sid in self.values:
                    val = self.values[sid].expected_value
                else:
                    # fallback: compute EV from outcomes or explicit value
                    if "value" in opt:
                        val = float(opt["value"])
                    else:
                        outs = opt.get("outcomes")
                        if outs:
                            val = float(sum([float(o.get("value", 0.0)) * float(o.get("prob", 0.0)) for o in outs]))
                        else:
                            val = 0.0
            scores.append(float(val))

        # convert to numpy
        arr = np.array(scores, dtype=float)
        chosen_idx = 0
        policy_used = policy.lower()

        if policy_used == "greedy":
            chosen_idx = int(np.argmax(arr))
        elif policy_used == "epsilon_greedy":
            if np.random.rand() < epsilon:
                chosen_idx = int(np.random.choice(len(options)))
            else:
                chosen_idx = int(np.argmax(arr))
        elif policy_used == "softmax":
            # softmax sampling with temperature
            if softmax_temp <= 0:
                chosen_idx = int(np.argmax(arr))
            else:
                # stable softmax
                shifted = arr - np.max(arr)
                expv = np.exp(shifted / softmax_temp)
                probs = expv / (np.sum(expv) + 1e-12)
                chosen_idx = int(np.random.choice(len(options), p=probs))
        else:
            # default greedy
            chosen_idx = int(np.argmax(arr))

        selected = options[chosen_idx]
        decision_id = f"ofc_decision_{int(time.time()*1000)}"
        details = {"scores": scores, "policy": policy_used, "chosen_index": chosen_idx}
        if persist_decision and self.db:
            try:
                self.db.save_decision(decision_id, selected.get("id", str(selected)), policy_used, details)
            except Exception:
                logger.exception("Failed to persist decision")

        # callback
        result = {"decision_id": decision_id, "chosen": selected, "scores": scores, "policy": policy_used}
        if self.on_decision:
            try:
                self.on_decision(result)
            except Exception:
                logger.exception("OFC on_decision hook failed")
        return result

    # ---------------------
    # Utilities: expected value with discount (future rewards)
    # ---------------------
    def compute_expected_value(self, action_id: str, future_rewards: Optional[List[float]] = None) -> float:
        """
        Compute immediate expected value for action_id plus discounted future rewards list.
        """
        immediate = float(self.evaluate_stimulus({"id": action_id}))
        if not future_rewards:
            return immediate
        disc = 0.0
        d = 1.0
        for r in future_rewards:
            d *= self.discount_factor
            disc += d * float(r)
        return float(immediate + disc)

    # ---------------------
    # Export / metrics
    # ---------------------
    def export_state(self) -> Dict[str, Any]:
        # sample top values
        vals = sorted(self.values.values(), key=lambda v: v.expected_value, reverse=True)[:10]
        top = [{"id": v.stimulus_id, "value": v.expected_value, "conf": v.confidence, "samples": len(v.history)} for v in vals]
        return {
            "system_id": self.system_id,
            "created_at": self.created_at,
            "n_values": len(self.values),
            "top_values": top,
            "learning_rate_base": self.base_lr,
            "discount_factor": self.discount_factor,
            "reversals_detected": self.reversals_detected,
            "decisions_made": self.decisions_made,
        }

    def recent_decisions(self, limit: int = 10):
        if self.db:
            return self.db.recent_decisions(limit)
        return []

# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    ofc = OrbitofrontalCortex(system_id="OFC-ENTERPRISE", persist=True, db_path="ofc_demo.db", base_learning_rate=0.25)

    # register hooks
    def on_update_hook(ve, meta):
        print("OFC UPDATE:", ve.stimulus_id, "EV=", ve.expected_value, "conf=", ve.confidence, "meta=", meta)

    def on_decision_hook(dec):
        print("OFC DECISION:", dec["decision_id"], "chosen=", dec["chosen"].get("id", dec["chosen"]))

    ofc.on_update = on_update_hook
    ofc.on_decision = on_decision_hook

    # define options with outcomes
    options = [
        {"id": "slot_A", "outcomes": [{"value": 10, "prob": 0.6}, {"value": -5, "prob": 0.4}]},
        {"id": "slot_B", "outcomes": [{"value": 2, "prob": 1.0}]},
        {"id": "slot_C", "outcomes": [{"value": 8, "prob": 0.3}, {"value": 1, "prob": 0.7}]}
    ]

    # initial evaluation (creates entries)
    for opt in options:
        ev = ofc.evaluate_stimulus(opt)
        print("Initial EV", opt["id"], ev)

    # simulate a sequence of outcomes and updates (reversal scenario)
    # T=0..4: slot_A good
    for t in range(5):
        chosen = ofc.choose_action(options, policy="epsilon_greedy", epsilon=0.0)  # greedy
        sid = chosen["chosen"]["id"]
        # simulate outcome: for first 5 trials slot_A yields +10 with prob 0.6 (here deterministic demonstration)
        outcome = 10.0 if sid == "slot_A" else 2.0
        print(f"Trial {t} chosen {sid} outcome {outcome}")
        pe_info = ofc.update_value(sid, outcome)
        print("PE info:", pe_info)

    # Now reversal: slot_A starts giving -10
    for t in range(5, 12):
        chosen = ofc.choose_action(options, policy="epsilon_greedy", epsilon=0.0)
        sid = chosen["chosen"]["id"]
        outcome = -10.0 if sid == "slot_A" else 2.0
        print(f"Trial {t} chosen {sid} outcome {outcome}")
        pe_info = ofc.update_value(sid, outcome)
        print("PE info:", pe_info)

    print("Final OFC state:")
    import pprint
    pprint.pprint(ofc.export_state())
    print("Recent decisions:", ofc.recent_decisions(10))
