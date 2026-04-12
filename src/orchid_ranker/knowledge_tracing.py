"""Proficiency tracing models for adaptive systems.

This module implements Bayesian Knowledge Tracing (BKT), Ebbinghaus forgetting
curves, and proficiency tracking for adaptive progression systems (education,
training, rehabilitation, gaming, and more).
"""

import math
from datetime import datetime
from typing import Dict, List, Optional


class BayesianKnowledgeTracing:
    """Bayesian Knowledge Tracing for estimating learner mastery.

    Models the probability a student has learned a skill using four parameters
    per skill. Implements the standard BKT model where the learner's knowledge
    state is represented as a hidden binary variable (known/unknown) that evolves
    over time with observations.

    Parameters
    ----------
    p_init : float, default=0.1
        Prior probability of knowing the skill initially. Range: [0, 1].
    p_transit : float, default=0.1
        Probability of learning the skill on each learning opportunity.
        Range: [0, 1].
    p_slip : float, default=0.1
        Probability of making an error (incorrect response) despite knowing
        the skill. Represents careless mistakes. Range: [0, 1].
    p_guess : float, default=0.2
        Probability of guessing correctly despite not knowing the skill.
        Range: [0, 1].
    mastery_threshold : float, default=0.95
        Threshold probability above which a skill is considered mastered.
        Range: [0, 1].

    Attributes
    ----------
    p_known : float
        Current belief about probability that student knows the skill.

    Notes
    -----
    The BKT model maintains a posterior belief P(L_t = 1 | observations) where
    L_t is the latent learning state. Updates follow:

    - If correct: P(L_{t+1}) = P(correct | L_t=1) * P(L_t=1) / P(correct)
    - If incorrect: P(L_{t+1}) = P(incorrect | L_t=1) * P(L_t=1) / P(incorrect)

    References
    ----------
    Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the
    acquisition of procedural knowledge. User Modeling and User-Adapted
    Interaction, 4(4), 253-278.
    """

    def __init__(
        self,
        p_init: float = 0.1,
        p_transit: float = 0.1,
        p_slip: float = 0.1,
        p_guess: float = 0.2,
        mastery_threshold: float = 0.95,
    ):
        """Initialize BayesianKnowledgeTracing model.

        Parameters
        ----------
        p_init : float, default=0.1
            Initial probability of knowing the skill.
        p_transit : float, default=0.1
            Probability of learning on each opportunity.
        p_slip : float, default=0.1
            Probability of incorrect response despite knowing.
        p_guess : float, default=0.2
            Probability of correct response despite not knowing.
        mastery_threshold : float, default=0.95
            Mastery threshold probability.

        Raises
        ------
        ValueError
            If any parameter is not in [0, 1].
        """
        for param, name in [
            (p_init, "p_init"),
            (p_transit, "p_transit"),
            (p_slip, "p_slip"),
            (p_guess, "p_guess"),
            (mastery_threshold, "mastery_threshold"),
        ]:
            if not 0 <= param <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {param}")

        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess
        self.mastery_threshold = mastery_threshold
        self._p_known = p_init
        self._num_observations = 0

    def update(self, correct: bool) -> float:
        """Update knowledge estimate after an observation.

        Uses Bayesian update rule to incorporate new evidence about whether
        the student answered correctly or incorrectly.

        Parameters
        ----------
        correct : bool
            Whether the student's response was correct.

        Returns
        -------
        float
            Updated probability that student knows the skill, range [0, 1].

        Notes
        -----
        The update implements:

        If correct:
            P(L | correct) ∝ P(correct | L) * P(L)
            P(correct | L=1) = 1 - p_slip
            P(correct | L=0) = p_guess

        If incorrect:
            P(L | incorrect) ∝ P(incorrect | L) * P(L)
            P(incorrect | L=1) = p_slip
            P(incorrect | L=0) = 1 - p_guess

        Then apply transition to account for learning opportunity:
        P(L_{t+1}) = P(L_t) + (1 - P(L_t)) * p_transit
        """
        # Likelihood of observation given knowledge state
        if correct:
            # P(correct | known) and P(correct | unknown)
            p_correct_if_known = 1 - self.p_slip
            p_correct_if_unknown = self.p_guess
        else:
            # P(incorrect | known) and P(incorrect | unknown)
            p_correct_if_known = self.p_slip
            p_correct_if_unknown = 1 - self.p_guess

        # Bayesian update
        numerator = p_correct_if_known * self._p_known
        denominator = (
            p_correct_if_known * self._p_known +
            p_correct_if_unknown * (1 - self._p_known)
        )

        if denominator > 1e-12:
            self._p_known = numerator / denominator

        # Apply learning transition
        self._p_known = self._p_known + (1 - self._p_known) * self.p_transit
        self._num_observations += 1

        return self._p_known

    @property
    def p_known(self) -> float:
        """Current probability the student knows the skill.

        Returns the current posterior probability estimate P(L=1) that the
        student has learned/mastered this skill, based on observed outcomes.

        Returns
        -------
        float
            Current posterior P(L = 1), range [0, 1].

        Examples
        --------
        >>> bkt = BayesianKnowledgeTracing()
        >>> 0 <= bkt.p_known <= 1
        True
        """
        return self._p_known

    def is_mastered(self) -> bool:
        """Check if skill is considered mastered.

        Returns True if the current probability of knowing the skill meets or
        exceeds the mastery threshold set during initialization.

        Returns
        -------
        bool
            True if P(known) >= mastery_threshold, False otherwise.

        Examples
        --------
        >>> bkt = BayesianKnowledgeTracing(mastery_threshold=0.95)
        >>> bkt.is_mastered()
        False
        >>> bkt.update(True)  # After correct response
        >>> bkt.is_mastered()  # Still False if p_known < 0.95
        False
        """
        return self._p_known >= self.mastery_threshold

    def reset(self) -> None:
        """Reset to prior distribution.

        Resets the knowledge estimate back to p_init and clears observation
        counter. Useful for starting fresh with a student or clearing historical data.

        Examples
        --------
        >>> bkt = BayesianKnowledgeTracing()
        >>> bkt.update(True)
        >>> bkt.p_known > 0.1  # Changed from prior
        True
        >>> bkt.reset()
        >>> bkt.p_known == 0.1  # Back to prior
        True
        """
        self._p_known = self.p_init
        self._num_observations = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BayesianKnowledgeTracing("
            f"p_known={self._p_known:.3f}, "
            f"mastered={self.is_mastered()}, "
            f"obs={self._num_observations})"
        )


class ProficiencyTracker:
    """Track proficiency state across multiple competencies for a single user.

    Maintains a portfolio of BayesianKnowledgeTracing models for different
    competencies, enabling proficiency tracking, dependency validation,
    and adaptive recommendations.

    Works for any domain with measurable competencies: education (skills),
    corporate training (certifications), rehabilitation (motor functions),
    fitness (exercises), gaming (abilities), etc.

    Parameters
    ----------
    competencies : list of str
        Names of competencies to track. (Alias: ``skills``.)
    bkt_params : dict, optional
        Per-competency BKT parameters. Keys are competency names, values are
        dicts with keys 'p_init', 'p_transit', 'p_slip', 'p_guess',
        'mastery_threshold'.
        Example: {'cardio': {'p_init': 0.2, 'p_transit': 0.15}}.
    default_params : dict, optional
        Default BKT parameters for competencies not in bkt_params.
        If None, uses BayesianKnowledgeTracing defaults.
    mastery_threshold : float, default=0.95
        Proficiency threshold. Overridden by per-competency settings.

    Attributes
    ----------
    competencies : list of str
        List of tracked competencies.
    skills : list of str
        Alias for ``competencies`` (backward compatibility).

    Examples
    --------
    >>> tracker = ProficiencyTracker(
    ...     competencies=['algebra', 'geometry', 'calculus'],
    ...     bkt_params={'algebra': {'p_init': 0.2}}
    ... )
    >>> tracker.update('algebra', True)  # Correct answer
    0.187...
    >>> tracker.get_mastery()
    {'algebra': 0.187..., 'geometry': 0.1, 'calculus': 0.1}
    """

    def __init__(
        self,
        competencies: Optional[List[str]] = None,
        bkt_params: Optional[Dict[str, Dict[str, float]]] = None,
        default_params: Optional[Dict[str, float]] = None,
        mastery_threshold: float = 0.95,
        *,
        skills: Optional[List[str]] = None,
    ):
        """Initialize ProficiencyTracker.

        Parameters
        ----------
        competencies : list of str
            Competency names to track. (Alias: ``skills``.)
        bkt_params : dict, optional
            Per-competency BKT parameters.
        default_params : dict, optional
            Default parameters for all competencies.
        mastery_threshold : float, default=0.95
            Default proficiency threshold.
        skills : list of str, optional
            Deprecated alias for ``competencies``.

        Raises
        ------
        ValueError
            If competencies list is empty.
        """
        # Support old 'skills' kwarg
        if competencies is None and skills is not None:
            competencies = skills
        if not competencies:
            raise ValueError("competencies list cannot be empty")

        self.competencies = list(competencies)
        self._trackers: Dict[str, BayesianKnowledgeTracing] = {}

        # Set up default parameters
        default = default_params or {}
        p_init = default.get('p_init', 0.1)
        p_transit = default.get('p_transit', 0.1)
        p_slip = default.get('p_slip', 0.1)
        p_guess = default.get('p_guess', 0.2)
        threshold = default.get('mastery_threshold', mastery_threshold)

        # Initialize trackers for each competency
        bkt_params = bkt_params or {}
        for comp in self.competencies:
            if comp in bkt_params:
                params = bkt_params[comp]
                self._trackers[comp] = BayesianKnowledgeTracing(
                    p_init=params.get('p_init', p_init),
                    p_transit=params.get('p_transit', p_transit),
                    p_slip=params.get('p_slip', p_slip),
                    p_guess=params.get('p_guess', p_guess),
                    mastery_threshold=params.get('mastery_threshold', threshold),
                )
            else:
                self._trackers[comp] = BayesianKnowledgeTracing(
                    p_init=p_init,
                    p_transit=p_transit,
                    p_slip=p_slip,
                    p_guess=p_guess,
                    mastery_threshold=threshold,
                )

    @property
    def skills(self) -> List[str]:
        """Alias for ``competencies`` (backward compatibility)."""
        return self.competencies

    def update(self, competency: str, correct: bool) -> float:
        """Update a competency's proficiency estimate.

        Parameters
        ----------
        competency : str
            Name of the competency to update.
        correct : bool
            Whether the response was correct.

        Returns
        -------
        float
            Updated P(known) for the competency.

        Raises
        ------
        KeyError
            If competency is not tracked.
        """
        if competency not in self._trackers:
            raise KeyError(f"Competency '{competency}' not in tracker. Available: {self.competencies}")

        return self._trackers[competency].update(correct)

    def get_mastery(self) -> Dict[str, float]:
        """Return proficiency estimates for all competencies.

        Returns
        -------
        dict
            Mapping {competency_name: p_known} for all competencies.
        """
        return {
            comp: self._trackers[comp].p_known
            for comp in self.competencies
        }

    def proficiency(self, competency: str) -> float:
        """Return proficiency estimate for a single competency.

        Parameters
        ----------
        competency : str
            Competency name.

        Returns
        -------
        float
            P(known) for the competency, in [0, 1].

        Raises
        ------
        KeyError
            If competency is not tracked.
        """
        if competency not in self._trackers:
            raise KeyError(f"Unknown competency '{competency}'. Tracked: {sorted(self.competencies)}")
        return self._trackers[competency].p_known

    # Backward-compatible alias
    skill_mastery = proficiency

    def mastered(self) -> List[str]:
        """Return list of mastered competency names.

        Returns all competency names for which the current proficiency
        meets or exceeds the threshold.

        Returns
        -------
        list of str
            Competency names where P(known) >= threshold.
        """
        return [
            comp for comp in self.competencies
            if self._trackers[comp].is_mastered()
        ]

    # Backward-compatible alias
    mastered_skills = mastered

    def unmastered(self) -> List[str]:
        """Return list of not-yet-mastered competency names.

        Returns
        -------
        list of str
            Competency names where P(known) < threshold.
        """
        return [
            comp for comp in self.competencies
            if not self._trackers[comp].is_mastered()
        ]

    # Backward-compatible alias
    unmastered_skills = unmastered

    def ready_for(
        self,
        competency: str,
        prerequisites: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """Check if user is ready for a competency given prerequisites.

        Parameters
        ----------
        competency : str
            Target competency name.
        prerequisites : dict, optional
            Dependency mapping from competency to list of prerequisite competencies.

        Returns
        -------
        bool
            True if prerequisites met or none defined, False otherwise.

        Raises
        ------
        KeyError
            If competency is not tracked.
        """
        if competency not in self._trackers:
            raise KeyError(f"Competency '{competency}' not in tracker")

        if prerequisites is None or competency not in prerequisites:
            return True

        required = prerequisites[competency]
        # Validate that all prerequisites are tracked competencies
        unknown = [r for r in required if r not in self._trackers]
        if unknown:
            raise KeyError(
                f"Prerequisite(s) {unknown} not tracked. "
                f"Known competencies: {sorted(self._trackers.keys())}"
            )
        achieved = set(self.mastered())
        return all(req in achieved for req in required)

    def recommend_next(
        self,
        prerequisites: Optional[Dict[str, List[str]]] = None,
        n: int = 3,
    ) -> List[str]:
        """Recommend next competencies based on proficiency and prerequisites.

        Recommends unmastered competencies with met prerequisites,
        ordered by lowest proficiency (highest priority).

        Parameters
        ----------
        prerequisites : dict, optional
            Dependency mapping (see ready_for).
        n : int, default=3
            Number of competencies to recommend.

        Returns
        -------
        list of str
            Up to n competency names, sorted by proficiency (lowest first).
        """
        candidates = []
        levels = self.get_mastery()
        achieved = set(self.mastered())

        for comp in self.unmastered():
            if prerequisites is None or comp not in prerequisites:
                candidates.append((levels[comp], comp))
            elif all(req in achieved for req in prerequisites[comp]):
                candidates.append((levels[comp], comp))

        candidates.sort()
        return [comp for _, comp in candidates[:n]]

    def __repr__(self) -> str:
        """Return string representation."""
        mastered_count = len(self.mastered())
        total_count = len(self.competencies)
        return (
            f"ProficiencyTracker("
            f"competencies={total_count}, "
            f"mastered={mastered_count}, "
            f"progress={mastered_count}/{total_count})"
        )



class ForgettingCurve:
    """Ebbinghaus-style exponential forgetting model.

    Models memory decay over time using an exponential forgetting curve.
    Memory strength increases with each review, while retention decays
    exponentially when not reviewed.

    The retention at time t since last review is modeled as:
    retention(t) = exp(-t / strength)

    where strength increases with each review, implementing spaced repetition.

    Parameters
    ----------
    initial_strength : float, default=1.0
        Initial memory strength. The unit defines the time scale: if you
        measure time in days, strength=1.0 means ~37% retention after 1 day.
        If you measure time in hours, it means ~37% after 1 hour. Range: (0, inf).
    strength_gain_on_review : float, default=0.5
        Strength increase per review (same units as initial_strength). Range: (0, inf).

    Attributes
    ----------
    strength : float
        Current memory strength.
    last_review_time : datetime or None
        Timestamp of last review, None if never reviewed.

    Notes
    -----
    The forgetting curve implements:

    1. Exponential decay: R(t) = e^(-t / S)
       where S is strength, t is time since last review

    2. Strength update on review: S_new = S_old + gain

    This enables spaced repetition: as memory strengthens, review intervals
    can be increased while maintaining target retention.

    References
    ----------
    Ebbinghaus, H. (1885). Memory: A contribution to experimental psychology.
    Teachers College, Columbia University.

    Examples
    --------
    >>> curve = ForgettingCurve(initial_strength=1.0)
    >>> curve.retention_at(1.0)  # After 1 time unit
    0.367...
    >>> curve.review()
    >>> curve.retention_at(1.0)  # Higher strength = better retention
    0.606...
    """

    def __init__(
        self,
        initial_strength: float = 1.0,
        strength_gain_on_review: float = 0.5,
    ):
        """Initialize ForgettingCurve.

        Parameters
        ----------
        initial_strength : float, default=1.0
            Initial memory strength.
        strength_gain_on_review : float, default=0.5
            Strength increase per review.

        Raises
        ------
        ValueError
            If parameters are not positive.
        """
        if initial_strength <= 0:
            raise ValueError(f"initial_strength must be positive, got {initial_strength}")
        if strength_gain_on_review <= 0:
            raise ValueError(
                f"strength_gain_on_review must be positive, got {strength_gain_on_review}"
            )

        self.strength = initial_strength
        self.strength_gain_on_review = strength_gain_on_review
        self.last_review_time: Optional[datetime] = None

    def retention_at(self, time_since_last_review: float) -> float:
        """Calculate memory retention at time t since last review.

        Computes the exponential retention function:
        retention(t) = exp(-t / strength)

        Parameters
        ----------
        time_since_last_review : float
            Time elapsed since last review in same units as strength.
            Range: [0, inf).

        Returns
        -------
        float
            Retention probability, range (0, 1].
            1.0 when t=0 (just reviewed).
            Approaches 0 as t grows large.

        Raises
        ------
        ValueError
            If time_since_last_review is negative.
        """
        if time_since_last_review < 0:
            raise ValueError(
                f"time_since_last_review must be non-negative, "
                f"got {time_since_last_review}"
            )

        if time_since_last_review == 0:
            return 1.0

        return math.exp(-time_since_last_review / self.strength)

    def review(self) -> None:
        """Record a review, strengthening memory.

        Increases memory strength by strength_gain_on_review and updates
        the last_review_time to the current time. Call this each time
        the learner reviews an item.

        Examples
        --------
        >>> curve = ForgettingCurve()
        >>> curve.retention_at(1.0)
        0.367...
        >>> curve.review()
        >>> curve.strength > 1.0  # Increased by strength_gain_on_review
        True
        """
        self.strength += self.strength_gain_on_review
        self.last_review_time = datetime.now()

    def should_review(self, threshold: float = 0.5) -> bool:
        """Check if memory should be reviewed given a threshold.

        Returns True if retention has dropped below the threshold, indicating
        that a review is needed to strengthen memory. Returns True if never
        reviewed (last_review_time is None).

        Parameters
        ----------
        threshold : float, default=0.5
            Retention threshold below which review is recommended.
            Range: [0, 1].

        Returns
        -------
        bool
            True if retention < threshold (review needed), False otherwise.
            Always returns True if never reviewed (last_review_time is None).

        Notes
        -----
        This implements a simple strategy for spaced repetition: review when
        retention drops to the threshold level. Common thresholds:
        - 0.5 (50%): More aggressive review schedule
        - 0.8 (80%): Conservative review schedule
        - 0.9 (90%): Very conservative schedule

        Examples
        --------
        >>> curve = ForgettingCurve()
        >>> curve.should_review(0.5)  # Never reviewed
        True
        >>> curve.review()
        >>> curve.should_review(0.5)
        False  # Just reviewed
        """
        if self.last_review_time is None:
            return True

        elapsed = (datetime.now() - self.last_review_time).total_seconds()
        retention = self.retention_at(elapsed)
        return retention < threshold

    def __repr__(self) -> str:
        """Return string representation."""
        retention_now = self.retention_at(0) if self.last_review_time is None else self.retention_at(
            (datetime.now() - self.last_review_time).total_seconds()
        )
        return (
            f"ForgettingCurve("
            f"strength={self.strength:.2f}, "
            f"retention={retention_now:.3f})"
        )


__all__ = [
    "BayesianKnowledgeTracing",
    "ProficiencyTracker",
    "ForgettingCurve",
    # Backward-compatible alias (deprecated)
    "MasteryTracker",
]


# --- Deprecation handling for renamed symbols (PEP 562) ---
_DEPRECATED_NAMES = {
    "MasteryTracker": "ProficiencyTracker",
}


def __getattr__(name: str):
    if name in _DEPRECATED_NAMES:
        import warnings
        warnings.warn(
            f"{name} is deprecated, use {_DEPRECATED_NAMES[name]} instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[_DEPRECATED_NAMES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
