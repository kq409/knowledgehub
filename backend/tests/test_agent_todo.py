import pytest

from services.agent.todo import MAX_TODO_ITEMS, TodoStatus, TodoStore, TodoStoreError


def test_replace_builds_a_snapshot():
    store = TodoStore()
    store.replace(
        [
            {"id": "a", "content": "List papers", "status": "completed"},
            {"content": "Read ELLA", "status": "in_progress"},
        ]
    )

    snapshot = store.snapshot()
    assert len(snapshot) == 2
    assert snapshot[0]["id"] == "a"
    assert snapshot[0]["status"] == "completed"
    assert snapshot[1]["status"] == "in_progress"
    assert snapshot[1]["id"]  # auto-assigned


def test_at_most_one_in_progress():
    store = TodoStore()
    with pytest.raises(TodoStoreError, match="in_progress"):
        store.replace(
            [
                {"content": "One", "status": "in_progress"},
                {"content": "Two", "status": "in_progress"},
            ]
        )


def test_rejects_too_many_items():
    store = TodoStore()
    items = [
        {"content": f"Step {i}", "status": "pending"} for i in range(MAX_TODO_ITEMS + 1)
    ]
    with pytest.raises(TodoStoreError, match="at most"):
        store.replace(items)


def test_rejects_empty_content_and_bad_status():
    store = TodoStore()
    with pytest.raises(TodoStoreError, match="content"):
        store.replace([{"content": "  ", "status": "pending"}])
    with pytest.raises(TodoStoreError, match="status"):
        store.replace([{"content": "Go", "status": "doing"}])


def test_format_for_model_lists_statuses():
    store = TodoStore()
    store.replace(
        [{"id": "1", "content": "Search", "status": TodoStatus.pending.value}]
    )
    text = store.format_for_model()
    assert "[pending]" in text
    assert "Search" in text
