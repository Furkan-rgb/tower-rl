"""The training entry point, end to end against the fake port.

No emulator, no adb, no bridge: `train_session` takes the instances it trains
against, so the double never reaches a path a device run can take. What is under
test is the thing the developer actually starts - argument parsing, arm
construction, the interleaved block schedule, periodic evaluation and
checkpointing, the learning curve the run is read from, and the fleet: a session
on several instances, its per-actor account, and its staggered bring-up.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import train  # noqa: E402
from clone_session import CloneInstance  # noqa: E402
from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.actor import ActorConfig  # noqa: E402
from tower_rl.application.evaluator import evaluate  # noqa: E402
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.domain.episode import TerminationOutcome  # noqa: E402
from tower_rl.domain.features import StateFeatures  # noqa: E402
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.ports.run_port import RunPortError  # noqa: E402

#: Tensors this small spend their time handing work between threads rather than
#: computing: one thread runs the whole file about fifteen times faster.
torch.set_num_threads(1)

PROFILE = "fake-profile-v1"

#: A network narrow enough that the entry point can be exercised in seconds. At
#: production width every decision is a CPU forward pass and dominates the run;
#: what is under test here is the plumbing around the learner, not its capacity,
#: which the backbone contract suite covers.
SMALL_NETWORK = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

def arguments(run_dir: Path, **overrides: str) -> argparse.Namespace:
    """The real parser, so the entry point's own defaults and checks are used."""
    argv: list[str] = []
    settings = {
        "--budget-decisions": "150",
        "--block-decisions": "50",
        "--batch-size": "2",
        "--gradient-steps-per-decision": "0.2",
        "--warmup-sequences": "2",
        "--sequence-length": "6",
        # Exactly `history-length - 1`, which is what fills the window.
        "--stacked-burn-in": "3",
        "--history-length": "4",
        "--replay-capacity": "64",
        "--evaluate-every-episodes": "1",
        "--evaluation-episodes": "2",
        # A run this short would never close a hundred-episode window.
        "--collection-window-episodes": "2",
        "--checkpoint-every-episodes": "2",
        "--serial": "fake-0",
        "--max-quiet-game-ms": "4000",
        "--run-dir": str(run_dir),
    }
    settings.update(overrides)
    for flag, value in settings.items():
        argv += [flag, value]
    return train.parse_arguments(argv)


def environment(**overrides: Any) -> InstrumentedRunEnvironment:
    settings: dict[str, Any] = {"damage_per_second": 2.0}
    settings.update(overrides)
    return InstrumentedRunEnvironment(
        port=FakeRunPort(**settings),
        builder=RunStateBuilder(profile_id=PROFILE),
        # `frame_game_ms` is the standing 100 ms of M1B-E018.
        cadence=CadenceConfig(frame_game_ms=100.0, max_quiet_game_ms=4000),
    )


def fleet(count: int = 1, **fake: Any) -> list[train.ActorInstance]:
    """`count` independent fake instances, named as a fleet's instances are."""
    return [
        train.ActorInstance(serial=f"fake-{index}", environment=environment(**fake))
        for index in range(count)
    ]


def session(
    run_dir: Path, budget: str = "120", actors: int = 1, **fake: Any
) -> dict[str, Any]:
    overrides = {"--budget-decisions": budget}
    if actors > 1:
        overrides.update(
            {
                "--actors": str(actors),
                # A fleet addresses its instances through `CloneInstance`, so
                # the single-actor serial may not be named beside it.
                "--serial": CloneInstance(index=0).serial,
                # Mid-run evaluation would need an instance to itself.
                "--evaluate-every-episodes": "0",
            }
        )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        return train.train_session(
            arguments(run_dir, **overrides),
            fleet(actors, **fake),
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
        )


#: Long enough that the arm plays more than one episode, which is what closes
#: a window of the collection curve.
TRAINING_BUDGET = "300"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One short session, reused by several checks."""
    run_dir = tmp_path_factory.mktemp("runs")
    return session(run_dir, budget=TRAINING_BUDGET)


def test_the_backbone_trains_under_one_budget(trained: dict[str, Any]) -> None:
    arms = trained["arms"]

    assert [arm["backbone"] for arm in arms] == ["stacked-dqn"]
    for arm in arms:
        assert arm["decisions"] >= int(TRAINING_BUDGET), "the arm spends the budget"
        assert arm["episodes"] > 0
        assert arm["optimisation_steps"] > 0
        assert arm["sequences_accepted"] > 0
        assert arm["failed_episodes"] == 0


def test_evaluation_runs_without_exploration(tmp_path: Path) -> None:
    """The entry point's evaluation goes through `evaluate`, which forces zero."""
    seen: list[float] = []

    class RecordingPolicy:
        def initial_state(self) -> None:
            return None

        def act(
            self, features: StateFeatures, state: None, *, epsilon: float
        ) -> tuple[int, None]:
            seen.append(epsilon)
            return next(index for index, allowed in enumerate(features.mask) if allowed), None

    report = evaluate(
        environment(),
        RecordingPolicy(),
        episodes=2,
        profile_id=PROFILE,
        # Exploration asked for and refused: evaluation is measurement.
        actor_config=ActorConfig(epsilon=1.0),
    )

    assert seen and set(seen) == {0.0}
    assert report.valid_episodes == 2


def test_an_episode_the_port_refuses_does_not_abort_the_session(tmp_path: Path) -> None:
    report = session(tmp_path, refuse_episodes=frozenset({2, 3}))

    # Episode ordinals are consumed by evaluation episodes too, so one refusal
    # lands on collection and one on an evaluation. Neither may end the session.
    arm = report["arms"][0]
    assert arm["failed_episodes"] >= 1
    assert len(arm["evaluation_failures"]) >= 1
    assert arm["failed_episodes"] + len(arm["evaluation_failures"]) == 2
    # The budget is still spent and the curve still produced.
    assert arm["decisions"] >= 120
    assert arm["learning_curve"]


def test_an_ambiguous_advance_is_classified_and_the_session_continues(
    tmp_path: Path,
) -> None:
    # Ordinal 1 is the first collected episode; evaluation episodes take the
    # ordinals after it.
    report = session(tmp_path, ambiguous_advance_episodes=frozenset({1}))

    arm = report["arms"][0]
    assert arm["failed_episodes"] == 0, "the port answered; the episode did not"
    assert arm["episodes"] > 1 and arm["decisions"] >= 120
    # The pipeline failure is an invalid episode, counted rather than fatal.
    assert arm["valid_episodes"] < arm["episodes"]
    assert arm["invalid_episodes_by_reason"] == {
        TerminationOutcome.ACTION_PIPELINE_FAILED.value: 1
    }


def test_the_regime_the_run_is_pinned_to_is_what_the_defaults_say(tmp_path: Path) -> None:
    """The settings of the second training run, where the developer reads them.

    Pinned as a test because every one of them was chosen against a measured
    failure of the first run; a silent drift back would cost another run of
    device time to discover.
    """
    defaults = train.parse_arguments(["--run-dir", str(tmp_path)])

    assert defaults.gradient_steps_per_decision == 0.25
    assert defaults.batch_size == 8
    assert defaults.warmup_sequences == 100
    assert defaults.sequence_length == 80
    assert defaults.stacked_burn_in == defaults.history_length - 1 == 7
    assert defaults.n_step == 10
    assert defaults.discount == 0.99
    assert defaults.learning_rate == 1e-4
    assert defaults.target_ema_decay == 0.995
    assert (defaults.epsilon_start, defaults.epsilon_end) == (1.0, 0.05)
    assert defaults.epsilon_anneal_decisions == 10_000
    assert defaults.priority_alpha == 0.0, "importance weights of exactly one"
    assert defaults.replay_capacity == 4096
    assert defaults.collection_window_episodes == 100
    assert defaults.evaluate_every_episodes == 0, "no frequent mid-run evaluation"
    assert defaults.evaluation_episodes == 30
    # One episode of parameter lag: a fleet's actors act from copies of the
    # network, and refreshing every episode is what a single actor acting from
    # the learner itself has always done.
    assert defaults.parameter_sync_episodes == 1


def test_a_stacked_burn_in_too_short_for_the_window_is_refused(tmp_path: Path) -> None:
    """Checked before the device is touched, not an hour into collection."""
    with pytest.raises(SystemExit, match="cannot fill a window"):
        arguments(tmp_path, **{"--stacked-burn-in": "2"})


#: Two fake instances, which is the fleet arrangement a device run takes: every
#: block of collection uses every actor.
FLEET_BUDGET = "300"


@pytest.fixture(scope="module")
def fleet_trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    run_dir = tmp_path_factory.mktemp("fleet")
    return session(run_dir, budget=FLEET_BUDGET, actors=2)


def test_a_fleet_trains_the_backbone_under_one_budget(
    fleet_trained: dict[str, Any],
) -> None:
    """The arm still works, and spends its budget across both instances."""
    assert fleet_trained["actors"] == 2
    assert fleet_trained["actor_serials"] == ["fake-0", "fake-1"]
    assert fleet_trained["bring_up_failures"] == []

    for arm in fleet_trained["arms"]:
        assert arm["decisions"] >= int(FLEET_BUDGET)
        assert arm["optimisation_steps"] > 0 and arm["sequences_accepted"] > 0
        assert arm["resolved_config"]["actors"] == 2
        assert arm["resolved_config"]["actor_ids"] == [
            f"fake-0:{arm['backbone']}",
            f"fake-1:{arm['backbone']}",
        ]
        # The pre-registered evaluation still lands, taken with the fleet stopped.
        assert arm["final_evaluation"]["pre_registered_final"] is True


def test_one_dead_instance_does_not_end_a_fleet_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One emulator refusing every episode costs an actor, not the run."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        report = train.train_session(
            arguments(
                tmp_path,
                **{
                    "--budget-decisions": "150",
                    "--actors": "2",
                    "--serial": CloneInstance(index=0).serial,
                    "--evaluate-every-episodes": "0",
                },
            ),
            [
                train.ActorInstance(serial="fake-0", environment=environment()),
                train.ActorInstance(
                    serial="fake-1", environment=environment(refuse_to_start=True)
                ),
            ],
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
        )

    arm = report["arms"][0]
    dead = next(actor for actor in arm["actors"] if actor["actor_id"] == "fake-1:stacked-dqn")
    alive = next(actor for actor in arm["actors"] if actor["actor_id"] == "fake-0:stacked-dqn")
    assert dead["withdrawn"] is not None and dead["failed_episodes"] > 0
    assert arm["actors_withdrawn"] == 1
    assert alive["decisions"] == arm["decisions"] >= 150
    assert arm["final_evaluation"] is not None, "the run was still measured"
    # A withdrawal is invisible in the aggregate, so it is announced when it
    # happens, naming the instance that left and what took it out.
    announcement = next(
        line for line in capsys.readouterr().out.splitlines() if "withdrawn" in line
    )
    assert "fake-1:stacked-dqn" in announcement and dead["withdrawn"] in announcement


def test_a_single_actor_run_records_exactly_one_actor(tmp_path: Path) -> None:
    """The default, and the configuration the in-flight run is reproducible from."""
    assert train.parse_arguments(["--run-dir", str(tmp_path)]).actors == 1

    report = session(tmp_path)

    assert report["actors"] == 1 and report["actor_serials"] == ["fake-0"]
    arm = report["arms"][0]
    assert arm["resolved_config"]["actors"] == 1
    assert [actor["actor_id"] for actor in arm["actors"]] == ["fake-0:stacked-dqn"]
    assert arm["actors"][0]["decisions"] == arm["decisions"]


def test_a_fleet_refuses_an_instance_named_by_hand(tmp_path: Path) -> None:
    """--serial and --port configure one actor; a fleet is addressed by index."""
    with pytest.raises(SystemExit, match="CloneInstance"):
        arguments(tmp_path, **{"--actors": "2", "--serial": "fake-0"})


def test_a_fleet_refuses_mid_run_evaluation(tmp_path: Path) -> None:
    """Evaluation borrows an instance, and every instance is collecting."""
    with pytest.raises(SystemExit, match="instance to itself"):
        arguments(
            tmp_path,
            **{
                "--actors": "2",
                "--serial": CloneInstance(index=0).serial,
                "--evaluate-every-episodes": "1",
            },
        )


def test_a_run_needs_at_least_one_actor(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="at least one actor"):
        arguments(tmp_path, **{"--actors": "0"})


def test_a_fleet_brings_its_instances_up_one_at_a_time(tmp_path: Path) -> None:
    """Four cold boots at once is the one thing the fleet measurement broke on."""
    spans: list[tuple[str, float, float]] = []

    def open_instance(instance: CloneInstance) -> train.ActorInstance:
        started = time.monotonic()
        time.sleep(0.01)
        spans.append((instance.serial, started, time.monotonic()))
        return train.ActorInstance(serial=instance.serial, environment=environment())

    instances = [CloneInstance(index=index) for index in range(4)]

    ready, failures = train.bring_up_fleet(instances, open_instance)

    assert failures == []
    assert [item.serial for item in ready] == [item.serial for item in instances]
    # No bring-up began before the previous one had concluded.
    for (_, _, ended), (_, next_started, _) in zip(spans, spans[1:], strict=False):
        assert next_started >= ended


def test_an_instance_that_will_not_come_up_costs_one_actor(tmp_path: Path) -> None:
    """A failed bring-up must not stall the instances behind it."""

    def open_instance(instance: CloneInstance) -> train.ActorInstance:
        if instance.index == 1:
            raise RuntimeError("never left main_unavailable")
        return train.ActorInstance(serial=instance.serial, environment=environment())

    ready, failures = train.bring_up_fleet(
        [CloneInstance(index=index) for index in range(3)], open_instance
    )

    assert [item.serial for item in ready] == ["emulator-5556", "emulator-5560"]
    assert len(failures) == 1 and "emulator-5558" in failures[0]


class _FailingAdapter:
    """An adapter whose release reads a bridge that has stopped answering."""

    def release(self) -> None:
        raise RunPortError("the bridge could not report the run state")


class _RecordingAdapter:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _RecordingClient:
    def __init__(self, port: int) -> None:
        self.port = port
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_teardown_continues_past_a_failing_release_and_puts_every_instance_down(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Leaving an emulator running is a safety failure, not an inconvenience.

    The release reads the bridge, so on a client that had stopped answering it
    raised inside the teardown's own `finally` and skipped every later release
    and every instance teardown: four emulators were left running, twice.
    """
    clients = [_RecordingClient(5555 + index) for index in range(2)]
    surviving = _RecordingAdapter()
    opened = [(_FailingAdapter(), clients[0]), (surviving, clients[1])]
    instances = [CloneInstance(index=index) for index in range(3)]
    torn: list[str] = []

    def tear_down(instance: CloneInstance) -> None:
        torn.append(instance.serial)
        if instance.index == 0:
            raise RuntimeError("adb would not stop this one")

    train.tear_down_fleet(opened, instances, tear_down)  # type: ignore[arg-type]

    # Neither the failing release nor the failing teardown stopped the rest.
    assert surviving.released and all(client.closed for client in clients)
    assert torn == [instance.serial for instance in instances]
    printed = capsys.readouterr().out
    assert "release failed" in printed and "teardown failed" in printed


def test_a_fleet_that_will_not_come_up_at_all_is_refused(tmp_path: Path) -> None:
    def refuse(instance: CloneInstance) -> train.ActorInstance:
        raise RuntimeError("no snapshot")

    with pytest.raises(SystemExit, match="no instance of the fleet came up"):
        train.bring_up_fleet([CloneInstance(index=0)], refuse)


def test_an_instance_whose_bring_up_fails_is_still_torn_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bring-up that fails partway must not leave its emulator running.

    `main`'s `open_instance` used to append to `started` only once bring-up had
    succeeded, so an instance whose bring-up raised was never in the list
    `tear_down_fleet` sweeps: the emulator it may already have launched
    survived a clean `exit 0`. Registration now happens before the attempt, so
    this exercises `main` end to end - against fakes, never a device - to prove
    the failed instance is still handed to teardown.
    """
    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", str(tmp_path))
    expected = SimpleNamespace(profile_id=PROFILE, bridge_version="v1")
    monkeypatch.setattr(train, "compatibility", lambda build_dir: expected)
    monkeypatch.setattr(train, "prepare_pinned_snapshot", lambda renderer, cores: "snap")
    monkeypatch.setattr(train, "require_offline", lambda instance: None)
    monkeypatch.setattr(
        train,
        "connect",
        lambda serial, port, arguments, expected, opened: train.ActorInstance(
            serial=serial, environment=environment()
        ),
    )

    def fake_bring_up(
        instance: CloneInstance, renderer: str, *, deploy: Any, read_only: bool, cores: int
    ) -> str:
        if instance.index == 1:
            raise RuntimeError("cold boot never reached home")
        return "cold"

    monkeypatch.setattr(train, "bring_up", fake_bring_up)

    torn: list[str] = []

    def fake_tear_down(instance: CloneInstance) -> None:
        torn.append(instance.serial)

    # `tear_down_fleet`'s teardown callable is bound as a default parameter at
    # definition time, exactly as `main` calls it with none supplied, so the
    # spy has to replace that default rather than pass an explicit argument.
    monkeypatch.setattr(train.tear_down_fleet, "__defaults__", (fake_tear_down,))

    captured: dict[str, Any] = {}

    def fake_train_session(
        arguments: argparse.Namespace, instances: list[train.ActorInstance], **kwargs: Any
    ) -> dict[str, Any]:
        captured["instances"] = list(instances)
        captured["bring_up_failures"] = kwargs["bring_up_failures"]
        return {}

    monkeypatch.setattr(train, "train_session", fake_train_session)
    monkeypatch.setattr(sys, "argv", ["train.py", "--actors", "2", "--no-track"])

    exit_code = train.main()

    assert exit_code == 0
    # Both instances are torn down, including the one whose bring-up failed.
    assert torn == ["emulator-5556", "emulator-5558"]
    assert captured["bring_up_failures"] and "emulator-5558" in captured["bring_up_failures"][0]
    assert [item.serial for item in captured["instances"]] == ["emulator-5556"]
