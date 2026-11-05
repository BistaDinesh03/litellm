"""A team or key in an access group that grants stores can use those stores and no others."""

import base64
from collections.abc import Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm import Router
from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import cache_access_object, vector_store_access_check
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.vector_store_endpoints.utils import (
    can_user_access_vector_store,
    can_user_read_vector_store,
    filter_listable_vector_stores,
)
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

TOKEN: Final = "hashed-key"
TEAM_ID: Final = "team-t"
OTHER_TEAM_ID: Final = "owner-team"


class ProxyState:
    """The proxy globals the checks read: a real cache in front of a database that holds no groups."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        group_reads: AsyncMock | None = None,
        team_reads: AsyncMock | None = None,
        cache: UserApiKeyCache | MagicMock | None = None,
    ) -> None:
        self.cache: Final = cache if cache is not None else UserApiKeyCache()
        self.group_reads: Final = group_reads if group_reads is not None else AsyncMock(return_value=None)
        self.prisma: Final = MagicMock(
            db=MagicMock(
                litellm_accessgrouptable=MagicMock(find_unique=self.group_reads),
                litellm_teamtable=MagicMock(find_unique=team_reads or AsyncMock(return_value=None)),
            )
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", self.prisma)
        monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", self.cache)

    async def add_group(
        self,
        group_id: str,
        stores: Sequence[str] = (),
        *,
        models: Sequence[str] = (),
        team_ids: Sequence[str] = (TEAM_ID,),
        key_ids: Sequence[str] = (TOKEN,),
    ) -> None:
        await cache_access_object(
            access_group_id=group_id,
            access_group_table=LiteLLM_AccessGroupTable(
                access_group_id=group_id,
                access_group_name=group_id,
                access_model_names=list(models),
                access_vector_store_ids=tuple(stores),
                assigned_team_ids=list(team_ids),
                assigned_key_ids=list(key_ids),
            ),
            user_api_key_cache=self.cache,
        )

    async def add_team(self, team: LiteLLM_TeamTableCachedObj) -> None:
        await self.cache.async_set_cache(key=f"team_id:{team.team_id}", value=team)


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> ProxyState:
    return ProxyState(monkeypatch)


def _permission(stores: Sequence[str] | None) -> LiteLLM_ObjectPermissionTable | None:
    if stores is None:
        return None
    return LiteLLM_ObjectPermissionTable(object_permission_id="own", vector_stores=list(stores))


def _key(
    own_stores: Sequence[str] | None = None, groups: Sequence[str] = (), team_id: str | None = None
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token=TOKEN,
        team_id=team_id,
        access_group_ids=list(groups),
        object_permission_id="own" if own_stores is not None else None,
        object_permission=_permission(own_stores),
        user_role=LitellmUserRoles.INTERNAL_USER,
    )


def _virtual_key(groups: Sequence[str]) -> UserAPIKeyAuth:
    key: Final = UserAPIKeyAuth(token=TOKEN, api_key=TOKEN, team_id=TEAM_ID, access_group_ids=list(groups))
    key.via_virtual_key = True
    return key


def _team(own_stores: Sequence[str] | None = None, groups: Sequence[str] = ()) -> LiteLLM_TeamTableCachedObj:
    return LiteLLM_TeamTableCachedObj(
        team_id=TEAM_ID,
        access_group_ids=list(groups),
        object_permission_id="own" if own_stores is not None else None,
        object_permission=_permission(own_stores),
    )


def _team_owned_store(store_id: str) -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(vector_store_id=store_id, custom_llm_provider="openai", team_id=OTHER_TEAM_ID)


async def _request_is_allowed(
    key: UserAPIKeyAuth, team: LiteLLM_TeamTableCachedObj | None, store: str, *, deny_by_default: bool = False
) -> bool:
    try:
        await vector_store_access_check(
            request_body={"vector_store_ids": [store]},
            team_object=team,
            valid_token=key,
            deny_by_default=deny_by_default,
        )
    except ProxyException:
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_stores", "key_groups", "team_stores", "team_groups", "store", "allowed"),
    [
        pytest.param(None, ("g1",), None, (), "store-a", True, id="key-group-grants"),
        pytest.param(None, ("g1",), None, (), "store-b", False, id="key-group-restricts"),
        pytest.param((), ("g1",), None, (), "store-b", False, id="key-group-restricts-with-empty-own-list"),
        pytest.param(None, (), None, ("g1",), "store-a", True, id="team-group-grants"),
        pytest.param(None, (), None, ("g1",), "store-b", False, id="team-group-restricts"),
        pytest.param(None, (), (), ("g1",), "store-b", False, id="team-group-restricts-with-empty-own-list"),
        pytest.param(("store-x",), ("g1",), None, (), "store-a", True, id="key-group-widens-own-list"),
        pytest.param(("store-x",), ("g1",), None, (), "store-x", True, id="key-own-list-still-counts"),
        pytest.param(("store-x",), ("g1",), None, (), "store-b", False, id="key-neither-list-denies"),
        pytest.param(None, (), ("store-x",), ("g1",), "store-a", True, id="team-group-widens-own-list"),
        pytest.param(None, (), ("store-x",), ("g1",), "store-x", True, id="team-own-list-still-counts"),
        pytest.param(("store-x",), (), None, ("g1",), "store-a", False, id="key-own-list-narrows-team-group"),
        pytest.param(None, ("g1",), ("store-x",), (), "store-a", False, id="team-own-list-narrows-key-group"),
        pytest.param(None, ("g1", "g2"), None, (), "store-c", True, id="stores-from-a-second-group"),
        pytest.param(None, ("models-only",), None, (), "store-b", True, id="models-only-group-changes-nothing"),
        pytest.param(None, (), None, ("models-only",), "store-b", True, id="team-models-only-group-changes-nothing"),
        pytest.param(("store-x",), ("models-only",), None, (), "store-b", False, id="models-only-keeps-own-list"),
        pytest.param(None, (), None, (), "store-b", True, id="no-grants-anywhere-changes-nothing"),
    ],
)
async def test_request_check_combines_own_list_and_group_stores(
    proxy: ProxyState,
    key_stores: Sequence[str] | None,
    key_groups: Sequence[str],
    team_stores: Sequence[str] | None,
    team_groups: Sequence[str],
    store: str,
    allowed: bool,
) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_group("g2", ("store-c",))
    await proxy.add_group("models-only", models=("gpt-4",))

    assert (
        await _request_is_allowed(_key(key_stores, key_groups, TEAM_ID), _team(team_stores, team_groups), store)
        is allowed
    )


@pytest.mark.asyncio
async def test_key_group_counts_when_it_lists_only_the_keys_team(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), key_ids=())

    assert await _request_is_allowed(_key(groups=("g1",), team_id=TEAM_ID), None, "store-a") is True
    assert await _request_is_allowed(_key(groups=("g1",), team_id=TEAM_ID), None, "store-b") is False


@pytest.mark.asyncio
async def test_key_cannot_claim_a_group_that_does_not_list_it(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), team_ids=(), key_ids=())

    assert await _request_is_allowed(_key(("store-x",), ("g1",)), None, "store-a") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store", "allowed"),
    [pytest.param("store-a", True, id="granted"), pytest.param("store-b", False, id="not-granted")],
)
async def test_deny_by_default_counts_group_grants(proxy: ProxyState, store: str, allowed: bool) -> None:
    await proxy.add_group("g1", ("store-a",))
    key: Final = _virtual_key(("g1",))

    assert await _request_is_allowed(key, _team(groups=("g1",)), store, deny_by_default=True) is allowed


@pytest.mark.asyncio
async def test_deny_by_default_requires_the_key_too_not_only_its_team(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    key: Final = _virtual_key(())

    assert await _request_is_allowed(key, _team(groups=("g1",)), "store-a", deny_by_default=True) is False


@pytest.mark.asyncio
async def test_managed_store_id_in_a_group_matches_its_provider_id(proxy: ProxyState) -> None:
    managed_id: Final = base64.urlsafe_b64encode(
        b"litellm_proxy:vector_store;unified_id,abc;target_model_names,gpt-4;resource_id,vs_provider"
    ).decode()
    await proxy.add_group("g1", (managed_id,))

    assert await _request_is_allowed(_key(groups=("g1",)), None, "vs_provider") is True


@pytest.mark.asyncio
async def test_keyless_team_member_gets_the_teams_group_stores(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    member: Final = UserAPIKeyAuth(team_id=TEAM_ID)

    assert await _request_is_allowed(member, _team(groups=("g1",)), "store-a") is True
    assert await _request_is_allowed(member, _team(groups=("g1",)), "store-b") is False


@pytest.mark.asyncio
async def test_deny_by_default_still_denies_a_models_only_group(proxy: ProxyState) -> None:
    await proxy.add_group("models-only", models=("gpt-4",))
    key: Final = _virtual_key(("models-only",))

    assert await _request_is_allowed(key, _team(groups=("models-only",)), "store-a", deny_by_default=True) is False


@pytest.mark.asyncio
async def test_unreadable_group_blocks_the_request_instead_of_lifting_the_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    with pytest.raises(ProxyException) as denied:
        await vector_store_access_check(
            request_body={"vector_store_ids": ["store-b"]},
            team_object=_team(groups=("g1",)),
            valid_token=_key(team_id=TEAM_ID),
        )

    assert denied.value.code == "503"
    assert denied.value.message == "Could not read an access group to check vector store access"


@pytest.mark.asyncio
async def test_deleted_group_grants_nothing_and_does_not_fail_the_request(proxy: ProxyState) -> None:
    assert await _request_is_allowed(_key(team_id=TEAM_ID), _team(groups=("deleted",)), "store-b") is True
    assert (
        await _request_is_allowed(_key(("store-x",), team_id=TEAM_ID), _team(groups=("deleted",)), "store-b") is False
    )


@pytest.mark.asyncio
async def test_group_cached_before_the_upgrade_keeps_the_callers_own_list(proxy: ProxyState) -> None:
    await proxy.cache.async_set_cache(
        key="access_group_id:old",
        value={"access_group_id": "old", "access_group_name": "old", "assigned_team_ids": [TEAM_ID]},
    )

    assert await _request_is_allowed(_key(("store-x",), team_id=TEAM_ID), _team(groups=("old",)), "store-b") is False
    assert await _request_is_allowed(_key(("store-x",), team_id=TEAM_ID), _team(groups=("old",)), "store-x") is True


@pytest.mark.asyncio
async def test_malformed_store_ids_are_rejected_when_only_a_group_restricts(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))

    with pytest.raises(ProxyException, match="vector_store_ids must be a list of strings"):
        await vector_store_access_check(
            request_body={"model": "gpt-4o-mini", "vector_store_ids": 123},
            team_object=_team(groups=("g1",)),
            valid_token=_key(team_id=TEAM_ID),
            llm_router=Router(model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}]),
        )


@pytest.mark.asyncio
async def test_request_naming_no_store_never_reads_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy: Final = ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    result: Final = await vector_store_access_check(
        request_body={"model": "gpt-4"}, team_object=_team(groups=("g1",)), valid_token=_key(team_id=TEAM_ID)
    )

    assert result is True
    assert proxy.group_reads.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_id", "allowed"),
    [pytest.param("store-a", True, id="group-store"), pytest.param("store-b", False, id="other-store")],
)
async def test_search_and_files_gate_counts_team_group_stores(proxy: ProxyState, store_id: str, allowed: bool) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(_team(groups=("g1",)))

    assert await can_user_read_vector_store(_team_owned_store(store_id), _key(team_id=TEAM_ID)) is allowed


@pytest.mark.asyncio
async def test_search_and_files_gate_counts_a_personal_keys_group_stores(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))

    assert await can_user_read_vector_store(_team_owned_store("store-a"), _key(groups=("g1",))) is True
    assert await can_user_read_vector_store(_team_owned_store("store-b"), _key(groups=("g1",))) is False


@pytest.mark.asyncio
async def test_group_grant_does_not_allow_managing_a_store(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(_team(groups=("g1",)))

    assert await can_user_access_vector_store(_team_owned_store("store-a"), _key(team_id=TEAM_ID)) is False


@pytest.mark.asyncio
async def test_models_only_group_grants_nothing_on_search_and_list(proxy: ProxyState) -> None:
    await proxy.add_group("models-only", models=("gpt-4",))
    await proxy.add_team(_team(groups=("models-only",)))

    assert await can_user_read_vector_store(_team_owned_store("store-a"), _key(team_id=TEAM_ID)) is False
    assert await filter_listable_vector_stores((_team_owned_store("store-a"),), _key(team_id=TEAM_ID)) == ()


@pytest.mark.asyncio
async def test_direct_grant_never_reads_teams_or_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    team_reads: Final = AsyncMock(side_effect=RuntimeError("database unavailable"))
    proxy: Final = ProxyState(monkeypatch, team_reads=team_reads)

    assert await can_user_read_vector_store(_team_owned_store("store-a"), _key(("store-a",), team_id=TEAM_ID)) is True
    listed: Final = await filter_listable_vector_stores(
        (_team_owned_store("store-a"),), _key(("store-a",), team_id=TEAM_ID)
    )

    assert listed == (_team_owned_store("store-a"),)
    assert team_reads.await_count == 0
    assert proxy.group_reads.await_count == 0


@pytest.mark.asyncio
async def test_dashboard_team_is_not_looked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    team_reads: Final = AsyncMock(return_value=None)
    ProxyState(monkeypatch, team_reads=team_reads)

    assert await can_user_read_vector_store(_team_owned_store("store-a"), _key(team_id=UI_TEAM_ID)) is False
    assert team_reads.await_count == 0


@pytest.mark.asyncio
async def test_list_shows_only_granted_stores(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(_team(groups=("g1",)))

    listed: Final = await filter_listable_vector_stores(
        (_team_owned_store("store-a"), _team_owned_store("store-b")), _key(team_id=TEAM_ID)
    )

    assert listed == (_team_owned_store("store-a"),)


@pytest.mark.asyncio
async def test_list_reads_the_group_once_however_many_stores_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = MagicMock(wraps=UserApiKeyCache())
    proxy: Final = ProxyState(monkeypatch, cache=cache)
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(_team(groups=("g1",)))

    await filter_listable_vector_stores(
        (_team_owned_store("store-a"), _team_owned_store("store-b"), _team_owned_store("store-c")),
        _key(team_id=TEAM_ID),
    )

    group_lookups: Final = [
        call for call in cache.async_get_cache.call_args_list if call.kwargs["key"] == "access_group_id:g1"
    ]
    assert len(group_lookups) == 1


@pytest.mark.asyncio
async def test_unreadable_group_grants_nothing_on_search_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy: Final = ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    await proxy.add_team(_team(groups=("g1",)))

    assert await can_user_read_vector_store(_team_owned_store("store-a"), _key(team_id=TEAM_ID)) is False


@pytest.mark.asyncio
async def test_failed_team_read_grants_nothing_and_keeps_direct_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    ProxyState(monkeypatch, team_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    assert await can_user_read_vector_store(_team_owned_store("store-a"), _key(team_id=TEAM_ID)) is False
    assert await can_user_read_vector_store(_team_owned_store("store-a"), _key(("store-a",), team_id=TEAM_ID)) is True
