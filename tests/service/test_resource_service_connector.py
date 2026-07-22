# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Core routing and task-state tests for Connector imports."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service import resource_service as resource_service_module
from openviking.service.resource_service import ResourceService
from openviking_cli.exceptions import InternalError, InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier

# Deterministic stand-in for the code-hosting predicate: routing tests must
# not depend on the real hosting-domain configuration.
_GIT_REPO_PREFIX = "https://git.example/"


class _BackgroundTask:
    def add_done_callback(self, _callback):
        pass


@pytest.fixture
def connector_config(monkeypatch):
    import openviking_cli.utils.config.open_viking_config as config_module

    config = SimpleNamespace(
        enable=True,
        connector="https://connector.example/doc/add",
        tracker="https://connector.example/task/info",
        timeout_seconds=60,
        poll_interval_ms=10,
        allowed_add_types=["tos"],
    )
    monkeypatch.setattr(
        config_module,
        "get_openviking_config",
        lambda: SimpleNamespace(connector=config),
    )
    monkeypatch.setattr(
        "openviking.connector.routing.is_git_repo_url",
        lambda path: isinstance(path, str) and path.startswith(_GIT_REPO_PREFIX),
    )
    return config


@pytest.fixture
def ctx():
    return RequestContext(
        user=UserIdentifier("acct", "alice"),
        role=Role.USER,
        api_key="secret",
    )


@pytest.fixture
def service():
    return ResourceService(
        vikingdb=object(),
        viking_fs=object(),
        resource_processor=object(),
        skill_processor=object(),
    )


def _task_tracker():
    return SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(task_id="task-1")),
        start=AsyncMock(),
        update_stage=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
    )


def _install_connector_dependencies(monkeypatch, tracker, connector_client):
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        resource_service_module,
        "ConnectorClient",
        lambda **_kwargs: connector_client,
    )

    def discard_monitor(coro):
        coro.close()
        return _BackgroundTask()

    monkeypatch.setattr(resource_service_module.asyncio, "create_task", discard_monitor)


@pytest.mark.asyncio
async def test_add_resource_routes_tos_to_connector(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"task_key": "connector-1"})
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    result = await service.add_resource(
        path="tos://bucket/a/b/c",
        ctx=ctx,
        parent="viking://resources/x/y",
    )

    assert result == {
        "status": "accepted",
        "task_id": "task-1",
        "connector_task_key": "connector-1",
        "resource_id": "viking://resources/x/y",
    }
    connector_client.submit_doc_add.assert_awaited_once_with(
        add_type="tos",
        api_key="secret",
        tos_path="bucket/a/b/c",
        path_prefix=["x", "y"],
        include_child=True,
        param_config=None,
        extra_params=None,
    )
    tracker.create.assert_awaited_once_with(
        "connector_import",
        resource_id="viking://resources/x/y",
        account_id="acct",
        user_id="alice",
    )


@pytest.mark.asyncio
async def test_add_resource_routes_git_repo_to_connector(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    connector_config.allowed_add_types = ["tos", "git"]
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"task_key": "connector-1"})
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    await service.add_resource(
        path="https://git.example/org/repo.git",
        ctx=ctx,
        parent="viking://resources/imports",
        args={"branch": "release"},
    )

    connector_client.submit_doc_add.assert_awaited_once_with(
        add_type="git",
        api_key="secret",
        tos_path=None,
        path_prefix=["imports"],
        include_child=True,
        param_config={
            "repo_url": "https://git.example/org/repo.git",
            "branch": "release",
        },
        extra_params=None,
    )


@pytest.mark.asyncio
async def test_git_connector_maps_ref_arg_to_branch(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    connector_config.allowed_add_types = ["git"]
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"task_key": "connector-1"})
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    await service.add_resource(
        path="https://git.example/org/repo",
        ctx=ctx,
        parent="viking://resources/imports",
        args={"ref": "v1.2"},
    )

    submitted = connector_client.submit_doc_add.await_args.kwargs
    assert submitted["add_type"] == "git"
    assert submitted["param_config"]["branch"] == "v1.2"
    assert submitted["param_config"]["repo_url"] == "https://git.example/org/repo"


@pytest.mark.asyncio
async def test_connector_import_persists_task_before_remote_submission(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    tracker = _task_tracker()

    async def fail_submission(**_kwargs):
        tracker.create.assert_awaited_once()
        raise RuntimeError("submission failed")

    connector_client = SimpleNamespace(submit_doc_add=AsyncMock(side_effect=fail_submission))
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    with pytest.raises(RuntimeError, match="submission failed"):
        await service.add_resource(
            path="tos://bucket/prefix",
            ctx=ctx,
            parent="viking://resources/imports",
        )

    tracker.fail.assert_awaited_once_with(
        "task-1",
        "submission failed",
        account_id="acct",
        user_id="alice",
    )


@pytest.mark.asyncio
async def test_connector_import_rejects_exact_to_target(
    connector_config,
    ctx,
    service,
):
    with pytest.raises(InvalidArgumentError, match="exact 'to' targets"):
        await service.add_resource(
            path="tos://bucket/a/b/c",
            ctx=ctx,
            to="viking://resources/x/y",
        )


def test_connector_only_route_rejects_disabled_or_unsupported_requests(
    connector_config,
    service,
):
    assert service._should_use_connector("https://example.com/doc") is False

    with pytest.raises(InvalidArgumentError, match="args keys"):
        service._should_use_connector("tos://bucket/prefix", connector_args={"parser": "pdf"})

    connector_config.enable = False
    with pytest.raises(InvalidArgumentError, match="Connector integration"):
        service._should_use_connector("tos://bucket/prefix")


def test_git_route_degrades_when_disabled_or_type_not_allowed(connector_config, service):
    # "git" not in allowed_add_types: standard pipeline handles the repo.
    assert service._should_use_connector("https://git.example/org/repo.git") is False

    connector_config.allowed_add_types = ["tos", "git"]
    connector_config.enable = False
    assert service._should_use_connector("https://git.example/org/repo.git") is False


@pytest.mark.parametrize(
    "target",
    [
        {"to": "viking://resources/repo"},
        {"parent": "viking://user/alice/resources/manuals"},
    ],
)
def test_git_source_falls_back_for_unsupported_targets(
    connector_config,
    service,
    target,
):
    connector_config.allowed_add_types = ["tos", "git"]

    assert service._should_use_connector("https://git.example/org/repo.git", **target) is False


def test_git_source_falls_back_for_unsupported_args(connector_config, service):
    connector_config.allowed_add_types = ["tos", "git"]

    assert (
        service._should_use_connector(
            "https://git.example/org/repo.git",
            connector_args={"commit": "abc123"},
        )
        is False
    )


@pytest.mark.parametrize(
    "parent",
    ["viking://resources/manuals", "resources/manuals"],
)
def test_connector_route_accepts_public_parent(connector_config, ctx, service, parent):
    connector_config.allowed_add_types = ["tos", "git"]

    assert (
        service._should_use_connector(
            "https://git.example/org/repo.git",
            ctx=ctx,
            parent=parent,
        )
        is True
    )


@pytest.mark.asyncio
async def test_add_resource_falls_back_for_shared_source_with_exact_to(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    connector_config.allowed_add_types = ["tos", "git"]
    monkeypatch.setattr(resource_service_module, "is_git_repo_url", lambda _path: True)
    service._add_resource_via_connector = AsyncMock()
    service.enqueue_git_add_resource = AsyncMock(return_value={"root_uri": "standard-pipeline"})

    result = await service.add_resource(
        path="https://git.example/org/repo.git",
        ctx=ctx,
        to="viking://resources/repo",
    )

    assert result == {"root_uri": "standard-pipeline"}
    service._add_resource_via_connector.assert_not_awaited()
    service.enqueue_git_add_resource.assert_awaited_once()


@pytest.mark.asyncio
async def test_connector_import_without_target_keeps_resource_id_unset(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"TaskKey": "connector-1"})
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    result = await service._add_resource_via_connector(
        path="tos://bucket/prefix",
        ctx=ctx,
        parent=None,
    )

    assert "resource_id" not in result
    assert tracker.create.await_args.kwargs["resource_id"] is None


@pytest.mark.asyncio
async def test_connector_import_rejects_target_outside_public_resources_root(
    connector_config,
    ctx,
    service,
):
    with pytest.raises(InvalidArgumentError, match="public resources root"):
        await service.add_resource(
            path="tos://bucket/prefix",
            ctx=ctx,
            parent="viking://user/alice/resources/spec",
        )


@pytest.mark.asyncio
async def test_connector_import_rejects_unsupported_args(
    connector_config,
    ctx,
    service,
):
    with pytest.raises(InvalidArgumentError, match="args keys"):
        await service.add_resource(
            path="tos://bucket/prefix",
            ctx=ctx,
            parent="viking://resources/spec",
            args={"parser": "pdf"},
        )


@pytest.mark.asyncio
async def test_add_resource_accepts_reason_without_wait(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"task_key": "connector-1"})
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    result = await service.add_resource(
        path="tos://bucket/prefix",
        ctx=ctx,
        parent="viking://resources/spec",
        reason="track quarterly reports",
    )

    assert result["status"] == "accepted"


@pytest.mark.asyncio
async def test_connector_import_wait_returns_terminal_result(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"task_key": "connector-1"}),
        get_task_info=AsyncMock(return_value={"Status": "succeeded"}),
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(resource_service_module.asyncio, "sleep", no_sleep)

    result = await service.add_resource(
        path="tos://bucket/prefix",
        ctx=ctx,
        parent="viking://resources/imports",
        wait=True,
    )

    assert result["status"] == "completed"
    assert result["task_id"] == "task-1"
    assert result["connector_task_key"] == "connector-1"
    assert result["resource_id"] == "viking://resources/imports"
    tracker.complete.assert_awaited_once()
    tracker.fail.assert_not_awaited()


@pytest.mark.asyncio
async def test_connector_import_wait_raises_on_failure(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"task_key": "connector-1"}),
        get_task_info=AsyncMock(
            return_value={"status": "failed", "error_message": "source unavailable"}
        ),
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(resource_service_module.asyncio, "sleep", no_sleep)

    with pytest.raises(InternalError, match="Connector import failed"):
        await service.add_resource(
            path="tos://bucket/prefix",
            ctx=ctx,
            parent="viking://resources/imports",
            wait=True,
        )

    tracker.fail.assert_awaited_once()
    tracker.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_connector_reason_links_memory_after_success(
    monkeypatch,
    connector_config,
    ctx,
    service,
):
    tracker = _task_tracker()
    connector_client = SimpleNamespace(
        submit_doc_add=AsyncMock(return_value={"task_key": "connector-1"}),
        get_task_info=AsyncMock(return_value={"Status": "succeeded"}),
    )
    _install_connector_dependencies(monkeypatch, tracker, connector_client)

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(resource_service_module.asyncio, "sleep", no_sleep)
    link_service = SimpleNamespace(on_resource_added=AsyncMock(return_value={"memories": 1}))
    service._resource_memory_link_service = link_service

    result = await service.add_resource(
        path="tos://bucket/prefix",
        ctx=ctx,
        parent="viking://resources/imports",
        wait=True,
        reason="track quarterly reports",
    )

    link_service.on_resource_added.assert_awaited_once_with(
        ctx=ctx,
        resource_uri="viking://resources/imports",
        reason="track quarterly reports",
        source_name=None,
        timeout=None,
    )
    assert result["memory_linking"] == {"memories": 1}
    completion = tracker.complete.await_args.args[1]
    assert completion["memory_linking"] == {"memories": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_info", "expected_stage", "expected_error"),
    [
        ({"Status": "succeeded"}, "connector:succeeded", None),
        (
            {"status": "failed", "error_message": "source unavailable"},
            "connector:failed",
            "connector task failed: source unavailable",
        ),
    ],
)
async def test_monitor_connector_task_maps_terminal_status(
    monkeypatch,
    connector_config,
    ctx,
    task_info,
    expected_stage,
    expected_error,
):
    tracker = _task_tracker()
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker",
        lambda: tracker,
    )

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(resource_service_module.asyncio, "sleep", no_sleep)
    client = SimpleNamespace(get_task_info=AsyncMock(return_value=task_info))

    outcome = await ResourceService()._monitor_connector_task(
        client=client,
        connector_task_key="connector-1",
        ov_task_id="task-1",
        poll_interval_ms=1,
        timeout_seconds=1,
        ctx=ctx,
    )

    assert tracker.update_stage.await_args.args[1] == expected_stage
    if expected_error is None:
        assert outcome["status"] == "completed"
        tracker.complete.assert_awaited_once()
        tracker.fail.assert_not_awaited()
    else:
        assert outcome == {"status": "failed", "error": expected_error}
        assert tracker.fail.await_args.args[1] == expected_error
        tracker.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_monitor_connector_task_retries_transient_polling_error(
    monkeypatch,
    connector_config,
    ctx,
):
    tracker = _task_tracker()
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker",
        lambda: tracker,
    )

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(resource_service_module.asyncio, "sleep", no_sleep)
    client = SimpleNamespace(
        get_task_info=AsyncMock(
            side_effect=[
                httpx.ReadTimeout("temporary timeout"),
                {"Status": "succeeded"},
            ]
        )
    )

    await ResourceService()._monitor_connector_task(
        client=client,
        connector_task_key="connector-1",
        ov_task_id="task-1",
        poll_interval_ms=1,
        timeout_seconds=1,
        ctx=ctx,
    )

    assert client.get_task_info.await_count == 2
    tracker.complete.assert_awaited_once()
    tracker.fail.assert_not_awaited()


@pytest.mark.asyncio
async def test_monitor_connector_task_marks_cancelled_monitor_as_failed(
    monkeypatch,
    connector_config,
    ctx,
):
    tracker = _task_tracker()
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker",
        lambda: tracker,
    )

    async def cancelled_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(resource_service_module.asyncio, "sleep", cancelled_sleep)
    client = SimpleNamespace(get_task_info=AsyncMock())

    with pytest.raises(asyncio.CancelledError):
        await ResourceService()._monitor_connector_task(
            client=client,
            connector_task_key="connector-1",
            ov_task_id="task-1",
            poll_interval_ms=1,
            timeout_seconds=1,
            ctx=ctx,
        )

    tracker.fail.assert_awaited_once_with(
        "task-1",
        "background connector task monitoring cancelled",
        account_id="acct",
        user_id="alice",
    )
    tracker.complete.assert_not_awaited()
