import torch

from orchid_ranker.agents.recommender_agent import TwoTowerRecommender


def test_two_tower_native_scoring_infer_shape():
    num_users, num_items, dim = 4, 10, 6
    rec = TwoTowerRecommender(
        num_users=num_users,
        num_items=num_items,
        user_dim=dim,
        item_dim=dim,
        use_native_scoring=True,
        dp_cfg={"enabled": False},
        use_bootts=False,
        use_linucb=False,
    ).eval()
    rec.user_matrix = torch.randn(num_users, dim)
    item_matrix = torch.randn(num_items, dim)
    user_ids = torch.tensor([0], dtype=torch.long)
    item_ids = torch.arange(5, dtype=torch.long)
    state_vec = torch.zeros(1, 4)

    logits = rec.infer(
        user_vec=None,
        item_matrix=item_matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        state_vec=state_vec,
    )

    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()

