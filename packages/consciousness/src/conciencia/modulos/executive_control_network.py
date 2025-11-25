# -*- coding: utf-8 -*-
"""
EXECUTIVE CONTROL NETWORK - EXTENDED ENTERPRISE
===============================================

Versión extendida del Executive Control Network (ECN)
- DLPFC (Working Memory + Planning)
- PPC (Attention Control)
- aPFC (Meta-Control)

Mejoras principales:
- Working memory determinista con decay por tiempo, rehearsal, chunking y políticas de evicción.
- Planificación jerárquica con subplans, replanificación, rollback y timeouts por paso.
- Inhibición con recursos y gating (interfaz para Basal Ganglia).
- Meta-control que ajusta políticas (learning simple).
- Persistencia ligera (SQLite opcional) y serialización JSON.
- Callbacks para eventos importantes (hook MCP/orquestador).
- Métricas y endpoints internos (export_state).

Requisitos: Python 3.10+, numpy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import time
import uuid
import json
import math
import sqlite3
import os
import logging
import numpy as np
from collections import deque

logger = logging.getLogger("ecn_extended")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# --------------------------
# Persistence helper (optional)
# --------------------------
class SimpleDB:
    def __init__(self, path: str = "ecn.db"):
        self.path = path
        self.conn = None
        try:
            init = not os.path.exists(path)
            self.conn = sqlite3.connect(path, check_same_thread=False)
            if init:
                self._init()
        except Exception:
            logger.exception("No se pudo inicializar DB")

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            plan_id TEXT PRIMARY KEY,
            created_ts REAL,
            goal TEXT,
            payload TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS wm_items (
            item_id TEXT PRIMARY KEY,
            added_ts REAL,
            payload TEXT
        )
        """)
        self.conn.commit()

    def save_plan(self, plan_id: str, goal: str, payload: dict):
        if not self.conn:
            return
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO plans (plan_id, created_ts, goal, payload) VALUES (?, ?, ?, ?)",
                    (plan_id, time.time(), goal, json.dumps(payload)))
        self.conn.commit()

    def save_wm_item(self, item_id: str, payload: dict):
        if not self.conn:
            return
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO wm_items (item_id, added_ts, payload) VALUES (?, ?, ?)",
                    (item_id, time.time(), json.dumps(payload)))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# --------------------------
# Working Memory item & utilities
# --------------------------
@dataclass
class WorkingMemoryItem:
    item_id: str
    content: Any
    activation: float  # 0..1
    priority: float  # 0..1 (higher = more important)
    added_ts: float = field(default_factory=time.time)
    last_refresh_ts: float = field(default_factory=time.time)
    decay_rate_per_s: float = 0.02  # default decay per second

    def age(self) -> float:
        return time.time() - self.added_ts

    def refresh(self, rehearse_strength: float = 0.2):
        """Rehearsal increases activation (bounded)"""
        self.activation = float(min(1.0, self.activation + rehearse_strength))
        self.last_refresh_ts = time.time()

    def decay(self, dt_s: float):
        """Decay proportional to dt"""
        dec = self.decay_rate_per_s * dt_s
        self.activation = float(max(0.0, self.activation - dec))


# --------------------------
# Plan & SubPlan
# --------------------------
@dataclass
class PlanStep:
    step_id: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_s: Optional[float] = None
    created_ts: float = field(default_factory=time.time)
    completed: bool = False
    success_probability: float = 0.8  # prior estimate
    attempt_count: int = 0


@dataclass
class Plan:
    plan_id: str
    goal: str
    steps: List[PlanStep]
    parent_plan_id: Optional[str] = None
    created_ts: float = field(default_factory=time.time)
    current_step_index: int = 0
    completed: bool = False
    success_probability: float = 0.5
    last_update_ts: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def current_step(self) -> Optional[PlanStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> Optional[PlanStep]:
        cs = self.current_step()
        if cs:
            cs.completed = True
            self.current_step_index += 1
            self.last_update_ts = time.time()
            if self.current_step_index >= len(self.steps):
                self.completed = True
            return cs
        return None

    def rollback(self, steps: int = 1):
        self.current_step_index = max(0, self.current_step_index - steps)
        self.completed = False
        self.last_update_ts = time.time()

    def estimate_success(self) -> float:
        """Estimación simple combinando step probabilities"""
        probs = [s.success_probability for s in self.steps]
        if not probs:
            return 0.0
        # multiplicative estimator (simplified)
        p = 1.0
        for pr in probs:
            p *= pr
        self.success_probability = float(max(0.0, min(1.0, p)))
        return self.success_probability


# --------------------------
# DLPFC: Working Memory & Planning
# --------------------------
class DorsolateralPFC:
    def __init__(self, capacity: int = 7, chunking_enabled: bool = True, db: Optional[SimpleDB] = None):
        self.capacity = capacity
        self.chunking_enabled = chunking_enabled
        self.wm: List[WorkingMemoryItem] = []
        self.plans: Dict[str, Plan] = {}
        self.db = db
        # metrics
        self.total_added = 0
        self.total_evicted = 0

    # -- Working Memory API --

    def _evict_policy(self) -> Optional[int]:
        """Decide index to evict: lowest (activation * priority) and oldest tie-breaker"""
        if not self.wm:
            return None
        scores = [(i, it.activation * it.priority, it.added_ts) for i, it in enumerate(self.wm)]
        # sort by score asc (lowest first), tie by oldest
        scores_sorted = sorted(scores, key=lambda x: (x[1], x[2]))
        return scores_sorted[0][0]

    def add_item(self, content: Any, priority: float = 0.5, rehearse: bool = False) -> WorkingMemoryItem:
        itm = WorkingMemoryItem(item_id=f"wm_{self.total_added}_{uuid.uuid4().hex[:6]}",
                                content=content,
                                activation=1.0,
                                priority=float(np.clip(priority, 0.0, 1.0)))
        # If similar item exists (simple content equality) rehearse instead of duplicate
        for existing in self.wm:
            if existing.content == content:
                existing.refresh(rehearse_strength=0.4)
                return existing

        if len(self.wm) >= self.capacity:
            idx = self._evict_policy()
            if idx is not None:
                evicted = self.wm.pop(idx)
                self.total_evicted += 1
                logger.debug("Evicted WM item %s (act=%.3f pr=%.3f)", evicted.item_id, evicted.activation, evicted.priority)
        self.wm.append(itm)
        self.total_added += 1
        # persist optionally
        if self.db:
            try:
                self.db.save_wm_item(itm.item_id, {"content": itm.content})
            except Exception:
                logger.exception("DB save wm_item failed")
        return itm

    def refresh_item(self, item_id: str, rehearse_strength: float = 0.2) -> bool:
        for it in self.wm:
            if it.item_id == item_id:
                it.refresh(rehearse_strength=rehearse_strength)
                return True
        return False

    def tick_decay(self, dt_s: float):
        """Aplica decay a todos los items basado en dt_s"""
        for item in list(self.wm):
            item.decay(dt_s)
            # Remove if decayed below threshold
            if item.activation < 0.05:
                self.wm.remove(item)
                self.total_evicted += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "load": len(self.wm),
            "items": [{"id": i.item_id, "act": i.activation, "prio": i.priority} for i in self.wm],
            "total_added": self.total_added,
            "total_evicted": self.total_evicted
        }

    # -- Planning API --

    def create_plan(self, goal: str, steps: List[Dict[str, Any]], parent_plan_id: Optional[str] = None) -> Plan:
        plan_id = f"plan_{len(self.plans)}_{uuid.uuid4().hex[:6]}"
        # convert steps to PlanStep objects
        plan_steps = [
            PlanStep(step_id=f"{plan_id}_step_{i}", action=s.get("action", str(s)), params=s.get("params", {}),
                     timeout_s=s.get("timeout_s"), success_probability=s.get("success_probability", 0.85))
            for i, s in enumerate(steps)
        ]
        plan = Plan(plan_id=plan_id, goal=goal, steps=plan_steps, parent_plan_id=parent_plan_id)
        plan.estimate_success()
        self.plans[plan_id] = plan
        if self.db:
            try:
                self.db.save_plan(plan_id, goal, {"steps": [s.action for s in plan_steps]})
            except Exception:
                logger.exception("DB save plan failed")
        return plan

    def step_plans(self) -> List[Tuple[str, Optional[PlanStep]]]:
        """Avanza planes que estén activos. Retorna lista de (plan_id, step_executed)"""
        results = []
        for pid, plan in list(self.plans.items()):
            if plan.completed:
                continue
            step = plan.current_step()
            if not step:
                continue
            # check timeout
            if step.timeout_s and (time.time() - step.created_ts) > step.timeout_s:
                # fallback: mark as failed for this step, attempt retry by lowering success prob
                step.attempt_count += 1
                step.success_probability = max(0.1, step.success_probability * 0.7)
                logger.info("Plan %s step %s timed out, lowering success_probability to %.2f", pid, step.step_id, step.success_probability)
                # maybe rollback or replan: simple policy -> rollback 1 step
                plan.rollback(steps=1)
                results.append((pid, None))
                continue
            # "execute" step - here we just mark attempted; actual execution integration required
            step.attempt_count += 1
            # success heuristics: if activation in WM matches step params -> success more likely
            success_chance = step.success_probability
            # if step.params references wm item id and it's present, boost chance
            if "requires_wm_item" in step.params:
                req = step.params["requires_wm_item"]
                if any(it.item_id == req for it in self.wm):
                    success_chance = min(1.0, success_chance + 0.15)
            # Deterministic decision based on threshold > 0.5
            executed = None
            if success_chance >= 0.5:
                executed = plan.advance()
            else:
                # attempt failed -> lower prob and possibly replan
                step.success_probability = max(0.05, step.success_probability * 0.8)
            results.append((pid, executed))
        return results


# --------------------------
# Posterior Parietal Cortex (PPC)
# --------------------------
class PosteriorParietalCortex:
    def __init__(self):
        self.attention_map: Dict[str, float] = {}
        self.current_focus: Optional[str] = None
        self.attention_shifts: int = 0

    def orient(self, location: str, strength: float = 1.0):
        self.attention_map[location] = float(np.clip(strength, 0.0, 1.0))
        if self.current_focus is None or self.attention_map.get(self.current_focus, 0.0) < self.attention_map[location]:
            self.current_focus = location
            self.attention_shifts += 1

    def shift(self, to: str):
        if to in self.attention_map:
            self.attention_map[to] = min(1.0, self.attention_map[to] + 0.2)
        else:
            self.attention_map[to] = 0.5
        self.current_focus = to
        self.attention_shifts += 1

    def distribution(self) -> Dict[str, float]:
        total = sum(self.attention_map.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.attention_map.items()}


# --------------------------
# Anterior PFC (Meta-Control) extended
# --------------------------
class AnteriorPFC:
    def __init__(self):
        self.strategies: Dict[str, float] = {}  # effectiveness estimate
        self.current_strategy: Optional[str] = None
        self.strategy_switches: int = 0
        self.learning_rate = 0.25

    def evaluate(self, strategy: str, outcome: float):
        if strategy not in self.strategies:
            self.strategies[strategy] = 0.5
        self.strategies[strategy] = (1 - self.learning_rate) * self.strategies[strategy] + self.learning_rate * outcome

    def select(self) -> Optional[str]:
        if not self.strategies:
            return None
        best = max(self.strategies.items(), key=lambda x: x[1])[0]
        if best != self.current_strategy:
            self.current_strategy = best
            self.strategy_switches += 1
        return best

    def register_strategy(self, name: str, baseline: float = 0.5):
        self.strategies.setdefault(name, baseline)


# --------------------------
# Basal Ganglia style gating (interface)
# --------------------------
class SimpleGate:
    """
    Deterministic gate: permite o inhibe acciones basadas en umbral y recursos.
    Interfaz para el ECN; puede conectarse con un módulo BG real.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def allow(self, salience: float, cognitive_load: float) -> bool:
        # regla simple: si salience > threshold*(1 + cognitive_load) => allow
        effective_thresh = self.threshold * (1.0 + cognitive_load)
        return salience >= effective_thresh


# --------------------------
# Executive Control Network - Integrador
# --------------------------
class ExecutiveControlNetwork:
    def __init__(self, system_id: str, wm_capacity: int = 7, persist_db_path: Optional[str] = None):
        self.system_id = system_id
        self.created_ts = time.time()
        # components
        db = SimpleDB(persist_db_path) if persist_db_path else None
        self.dlpfc = DorsolateralPFC(capacity=wm_capacity, db=db)
        self.ppc = PosteriorParietalCortex()
        self.apfc = AnteriorPFC()
        self.gate = SimpleGate(threshold=0.55)
        # state
        self.cognitive_load = 0.0
        self.inhibition_active = False
        self.control_mode: str = "automatic"
        # metrics
        self.control_events = 0
        self.inhibition_events = 0
        self.plan_success_count = 0
        # callbacks
        self.on_new_plan: Optional[Callable[[Plan], None]] = None
        self.on_step_completed: Optional[Callable[[Plan, PlanStep], None]] = None
        self.on_inhibition: Optional[Callable[[Dict[str, Any]], None]] = None

    # Public API
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entrada: task dict con fields:
            - type, content, priority, location, conflict(bool), steps (optional), novelty(bool)
        """
        self.control_events += 1

        complexity = self._assess_complexity(task)
        self.control_mode = "controlled" if complexity > 0.5 else "automatic"

        # add to working memory if controlled
        if self.control_mode == "controlled":
            wm_item = self.dlpfc.add_item(content=task, priority=task.get("priority", 0.5))
        else:
            wm_item = None

        # orient attention
        loc = task.get("location", "central")
        self.ppc.orient(loc, strength=complexity)

        # inhibition if conflict
        if task.get("conflict", False):
            self._inhibit(task)

        # create plan if provided
        plan = None
        if "steps" in task and isinstance(task["steps"], list):
            steps = [{"action": s, "params": {}} for s in task["steps"]]
            plan = self.create_plan(task.get("goal", "auto_goal"), steps)

        # update cognitive load
        self.cognitive_load = len(self.dlpfc.wm) / max(1, self.dlpfc.capacity)

        return {
            "mode": self.control_mode,
            "wm_item": wm_item.item_id if wm_item else None,
            "attention_focus": self.ppc.current_focus,
            "plan_id": plan.plan_id if plan else None,
            "can_process": self.cognitive_load < 0.95
        }

    def step(self, dt_s: float = 1.0):
        """
        Tick method to be called periodically by the orchestrator.
        - decays WM items
        - advances plans
        - updates metrics
        """
        self.dlpfc.tick_decay(dt_s)
        results = self.dlpfc.step_plans()
        for pid, step in results:
            if step:
                # step executed (deterministically)
                # notify
                if self.on_step_completed:
                    try:
                        self.on_step_completed(self.dlpfc.plans[pid], step)
                    except Exception:
                        logger.exception("on_step_completed callback error")
                # update metrics
                if self.dlpfc.plans[pid].completed:
                    self.plan_success_count += 1
            else:
                # step timed out or failed -> could notify
                pass

        # recalc cognitive load
        self.cognitive_load = len(self.dlpfc.wm) / max(1, self.dlpfc.capacity)

    def create_plan(self, goal: str, steps: List[Dict[str, Any]]) -> Plan:
        plan = self.dlpfc.create_plan(goal, steps)
        if self.on_new_plan:
            try:
                self.on_new_plan(plan)
            except Exception:
                logger.exception("on_new_plan callback error")
        return plan

    def interrupt(self, reason: str, priority: float = 0.9):
        """
        Interruption: push high-priority item and optionally pre-empt plans
        """
        self.inhibition_active = True
        self.inhibition_events += 1
        interrupt_task = {"type": "interrupt", "desc": reason, "priority": priority}
        self.dlpfc.add_item(interrupt_task, priority=priority)
        if self.on_inhibition:
            try:
                self.on_inhibition({"reason": reason, "priority": priority})
            except Exception:
                logger.exception("on_inhibition callback error")

    def _inhibit(self, task: Dict[str, Any]):
        # decide via gate
        salience = task.get("priority", 0.5)
        allow = self.gate.allow(salience, self.cognitive_load)
        if not allow:
            self.inhibition_active = True
            self.inhibition_events += 1
            if self.on_inhibition:
                try:
                    self.on_inhibition({"task": task})
                except Exception:
                    logger.exception("on_inhibition callback error")
        else:
            self.inhibition_active = False

    def _assess_complexity(self, task: Dict[str, Any]) -> float:
        c = 0.0
        if task.get("novel", False):
            c += 0.35
        if task.get("conflict", False):
            c += 0.4
        if "steps" in task:
            c += min(0.5, 0.1 * len(task["steps"]))
        return min(1.0, c)

    def get_executive_state(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "created": self.created_ts,
            "cognitive_load": self.cognitive_load,
            "inhibition_active": self.inhibition_active,
            "control_mode": self.control_mode,
            "dlpfc": self.dlpfc.get_state(),
            "ppc": self.ppc.distribution(),
            "aPFC": {"current_strategy": self.apfc.current_strategy, "strategies": self.apfc.strategies},
            "metrics": {
                "control_events": self.control_events,
                "inhibition_events": self.inhibition_events,
                "plans_success": self.plan_success_count
            }
        }


# --------------------------
# Ejemplo de uso / tests rápidos
# --------------------------
if __name__ == "__main__":
    ecn = ExecutiveControlNetwork("ECN-ENTERPRISE", wm_capacity=7, persist_db_path=None)

    # Callbacks
    def on_new_plan(plan: Plan):
        print("NEW PLAN:", plan.plan_id, "goal:", plan.goal)

    def on_step_completed(plan: Plan, step: PlanStep):
        print("STEP COMPLETED:", plan.plan_id, "step:", step.step_id, "action:", step.action)

    def on_inh(info: Dict[str, Any]):
        print("INHIBITION EVENT:", info)

    ecn.on_new_plan = on_new_plan
    ecn.on_step_completed = on_step_completed
    ecn.on_inhibition = on_inh

    # Simulate tasks
    t1 = {"type": "read_email", "content": "email A", "priority": 0.3, "location": "inbox", "steps": ["open", "read", "archive"], "novel": False}
    t2 = {"type": "stop_intruder", "content": "intrusion", "priority": 0.95, "location": "door", "conflict": True, "novel": True,
          "steps": [{"action": "assess"}, {"action": "call_security"}, {"action": "block_exit"}]}

    print("Process task 1:", ecn.process_task(t1))
    print("Process task 2:", ecn.process_task(t2))

    # Tick some times
    for i in range(5):
        ecn.step(dt_s=1.0)
        time.sleep(0.01)

    # Create manual plan
    p = ecn.create_plan("deliver_message", [{"action": "compose"}, {"action": "send"}])
    print("Created plan:", p.plan_id, "estimated success:", p.estimate_success())

    # Simulate steps until completion
    while not p.completed and p.current_step_index < 10:
        ecn.step(dt_s=1.0)
        time.sleep(0.01)

    print("Final Executive State:")
    print(json.dumps(ecn.get_executive_state(), indent=2, ensure_ascii=False))
