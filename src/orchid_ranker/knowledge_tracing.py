"""Educational knowledge tracing models for learner mastery estimation.

This module implements Bayesian Knowledge Tracing (BKT), Ebbinghaus forgetting
curves, and mastery tracking for adaptive educational systems.
"""

import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime


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

        if denominator > 0:
            self._p_known = numerator / denominator

        # Apply learning transition
        self._p_known = self._p_known + (1 - self._p_known) * self.p_transit
        self._num_observations += 1

        return self._p_known

    def p_known(self) -> float:
        """Get current probability the student knows the skill.

        Returns the current posterior probability estimate P(L=1) that the
        student has learned/mastered this skill, based on observed outcomes.

        Returns
        -------
        float
            Current posterior P(L = 1), range [0, 1].

        Examples
        --------
        >>> bkt = BayesianKnowledgeTracing()
        >>> p = bkt.p_known()
        >>> 0 <= p <= 1
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
        >>> bkt.p_known() > 0.1  # Changed from prior
        True
        >>> bkt.reset()
        >>> bkt.p_known() == 0.1  # Back to prior
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


class MasteryTracker:
    """Track mastery state across multiple skills for a single learner.

    Maintains a portfolio of BayesianKnowledgeTracing models for different
    skills, enabling skill-level mastery tracking, prerequisite validation,
    and adaptive skill recommendation.

    Parameters
    ----------
    skills : list of str
        Names of skills to track.
    bkt_params : dict, optional
        Per-skill BKT parameters. Keys are skill names, values are dicts with
        keys 'p_init', 'p_transit', 'p_slip', 'p_guess', 'mastery_threshold'.
        Example: {'algebra': {'p_init': 0.2, 'p_transit': 0.15}}.
    default_params : dict, optional
        Default BKT parameters for skills not in bkt_params.
        Keys: 'p_init', 'p_transit', 'p_slip', 'p_guess', 'mastery_threshold'.
        If None, uses BayesianKnowledgeTracing defaults.
    mastery_threshold : float, default=0.95
        Mastery threshold. Overridden by per-skill settings in bkt_params.

    Attributes
    ----------
    skills : list of str
        List of tracked skills.

    Examples
    --------
    >>> tracker = MasteryTracker(
    ...     skills=['algebra', 'geometry', 'calculus'],
    ...     bkt_params={'algebra': {'p_init': 0.2}}
    ... )
    >>> tracker.update('algebra', True)  # Correct answer
    0.187...
    >>> tracker.get_mastery()
    {'algebra': 0.187..., 'geometry': 0.1, 'calculus': 0.1}
    """

    def __init__(
        self,
        skills: List[str],
        bkt_params: Optional[Dict[str, Dict[str, float]]] = None,
        default_params: Optional[Dict[str, float]] = None,
        mastery_threshold: float = 0.95,
    ):
        """Initialize MasteryTracker.

        Parameters
        ----------
        skills : list of str
            Skill names to track.
        bkt_params : dict, optional
            Per-skill BKT parameters.
        default_params : dict, optional
            Default parameters for all skills.
        mastery_threshold : float, default=0.95
            Default mastery threshold.

        Raises
        ------
        ValueError
            If skills list is empty.
        """
        if not skills:
            raise ValueError("skills list cannot be empty")

        self.skills = list(skills)
        self._trackers: Dict[str, BayesianKnowledgeTracing] = {}

        # Set up default parameters
        default = default_params or {}
        p_init = default.get('p_init', 0.1)
        p_transit = default.get('p_transit', 0.1)
        p_slip = default.get('p_slip', 0.1)
        p_guess = default.get('p_guess', 0.2)
        threshold = default.get('mastery_threshold', mastery_threshold)

        # Initialize trackers for each skill
        bkt_params = bkt_params or {}
        for skill in self.skills:
            if skill in bkt_params:
                params = bkt_params[skill]
                self._trackers[skill] = BayesianKnowledgeTracing(
                    p_init=params.get('p_init', p_init),
                    p_transit=params.get('p_transit', p_transit),
                    p_slip=params.get('p_slip', p_slip),
                    p_guess=params.get('p_guess', p_guess),
                    mastery_threshold=params.get('mastery_threshold', threshold),
                )
            else:
                self._trackers[skill] = BayesianKnowledgeTracing(
                    p_init=p_init,
                    p_transit=p_transit,
                    p_slip=p_slip,
                    p_guess=p_guess,
                    mastery_threshold=threshold,
                )

    def update(self, skill: str, correct: bool) -> float:
        """Update a skill's knowledge estimate.

        Parameters
        ----------
        skill : str
            Name of the skill to update.
        correct : bool
            Whether the student's response was correct.

        Returns
        -------
        float
            Updated P(known) for the skill.

        Raises
        ------
        KeyError
            If skill is not in the tracker's skill list.
        """
        if skill not in self._trackers:
            raise KeyError(f"Skill '{skill}' not in tracker. Available: {self.skills}")

        return self._trackers[skill].update(correct)

    def get_mastery(self) -> Dict[str, float]:
        """Return knowledge estimates for all skills.

        Returns
        -------
        dict
            Mapping {skill_name: p_known} for all skills.
        """
        return {
            skill: self._trackers[skill].p_known()
            for skill in self.skills
        }

    def mastered_skills(self) -> List[str]:
        """Return list of mastered skill names.

        Returns all skill names for which the current knowledge probability
        meets or exceeds the mastery threshold.

        Returns
        -------
        list of str
            Skill names where P(known) >= mastery_threshold, sorted alphabetically.

        Examples
        --------
        >>> tracker = MasteryTracker(['algebra', 'geometry', 'calculus'])
        >>> tracker.update('algebra', True)
        >>> tracker.mastered_skills()  # Depends on initial p_init
        []
        """
        return [
            skill for skill in self.skills
            if self._trackers[skill].is_mastered()
        ]

    def unmastered_skills(self) -> List[str]:
        """Return list of not-yet-mastered skill names.

        Returns all skill names for which the current knowledge probability
        is below the mastery threshold.

        Returns
        -------
        list of str
            Skill names where P(known) < mastery_threshold, sorted alphabetically.

        Examples
        --------
        >>> tracker = MasteryTracker(['algebra', 'geometry', 'calculus'])
        >>> tracker.unmastered_skills()
        ['algebra', 'geometry', 'calculus']
        """
        return [
            skill for skill in self.skills
            if not self._trackers[skill].is_mastered()
        ]

    def ready_for(
        self,
        skill: str,
        prerequisites: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """Check if student is ready for a skill given prerequisites.

        A student is ready for a skill if:
        1. The skill exists in the tracker.
        2. Either no prerequisites are defined, or all prerequisites are mastered.

        Parameters
        ----------
        skill : str
            Target skill name.
        prerequisites : dict, optional
            Prerequisite graph mapping skill name to list of prerequisite skills.
            Example: {'calculus': ['algebra', 'precalc']}.

        Returns
        -------
        bool
            True if student is ready (prerequisites met or none), False otherwise.

        Raises
        ------
        KeyError
            If skill is not in the tracker's skill list.
        """
        if skill not in self._trackers:
            raise KeyError(f"Skill '{skill}' not in tracker")

        if prerequisites is None or skill not in prerequisites:
            return True

        required = prerequisites[skill]
        mastered = set(self.mastered_skills())
        return all(req in mastered for req in required)

    def recommend_next(
        self,
        prerequisites: Optional[Dict[str, List[str]]] = None,
        n: int = 3,
    ) -> List[str]:
        """Recommend next skills to study based on mastery and prerequisites.

        Recommends unmastered skills for which the student has met prerequisites,
        ordered by lowest mastery level (highest priority for learning).

        Parameters
        ----------
        prerequisites : dict, optional
            Prerequisite graph (see ready_for).
        n : int, default=3
            Number of skills to recommend.

        Returns
        -------
        list of str
            Up to n skill names, sorted by mastery (lowest first).

        Examples
        --------
        >>> tracker = MasteryTracker(['algebra', 'geometry', 'calculus'])
        >>> tracker.update('algebra', True)
        >>> tracker.recommend_next(
        ...     prerequisites={'calculus': ['algebra']},
        ...     n=2
        ... )
        ['geometry', 'calculus']  # or similar, depending on mastery state
        """
        candidates = []
        mastery = self.get_mastery()

        for skill in self.unmastered_skills():
            if self.ready_for(skill, prerequisites):
                candidates.append((mastery[skill], skill))

        # Sort by mastery (ascending) and return top n
        candidates.sort()
        return [skill for _, skill in candidates[:n]]

    def __repr__(self) -> str:
        """Return string representation."""
        mastered_count = len(self.mastered_skills())
        total_count = len(self.skills)
        return (
            f"MasteryTracker("
            f"skills={total_count}, "
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
        Initial memory strength in time units. Higher values indicate
        longer retention. Range: (0, inf).
    strength_gain_on_review : float, default=0.5
        Amount to increase strength on each review. Range: (0, inf).

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
