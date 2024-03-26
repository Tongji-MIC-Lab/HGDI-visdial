import torch

def scores_to_ranks(scores: torch.Tensor):
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks


class SparseGTMetrics(object):
    def __init__(self):
        self._rank_list = []

    def observe(
        self, predicted_scores: torch.Tensor, target_ranks: torch.Tensor
    ):
        predicted_scores = predicted_scores.detach()

        predicted_ranks = scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()

        # collapse batch dimension
        predicted_ranks = predicted_ranks.view(
            batch_size * num_rounds, num_options
        )

        target_ranks = target_ranks.view(batch_size * num_rounds).long()

        predicted_gt_ranks = predicted_ranks[
            torch.arange(batch_size * num_rounds), target_ranks
        ]
        self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))

    def retrieve(self, reset: bool = True):
        num_examples = len(self._rank_list)
        if num_examples > 0:
            # convert to numpy array for easy calculation.
            __rank_list = torch.tensor(self._rank_list).float()
            metrics = {
                "r@1": torch.mean((__rank_list <= 1).float()).item(),
                "r@5": torch.mean((__rank_list <= 5).float()).item(),
                "r@10": torch.mean((__rank_list <= 10).float()).item(),
                "mean": torch.mean(__rank_list).item(),
                "mrr": torch.mean(__rank_list.reciprocal()).item(),
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._rank_list = []


class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
        self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        predicted_scores = predicted_scores.detach()

        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze()
        batch_size, num_options = predicted_ranks.size()

        k = torch.sum(target_relevance != 0, dim=-1)

        _, rankings = torch.sort(predicted_ranks, dim=-1)
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True):
        if self._ndcg_denominator > 0:
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0
