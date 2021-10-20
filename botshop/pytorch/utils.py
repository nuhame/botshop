import torch
import torch.nn.functional as F


def random_sample(logits, T):
    p = F.softmax(logits / T, dim=0)

    token_idx = torch.multinomial(p.view(-1), 1)
    token_p = torch.gather(p, 0, token_idx)

    return token_p.squeeze(), token_idx.squeeze()


def select_max(logits):
    p = F.softmax(logits, dim=0)

    token_p, token_idx = torch.max(p, dim=0)

    return token_p, token_idx


def top_p_sample(p, top_p=0.9, temperature=1.0):
    """

    TODO : make it work for batch_size > 0

    :param p: shape (1, batch_size, num_tokens)
    :param top_p:
    :param temperature:
    :return:
    """

    p_sorted, sorted_indices = torch.sort(p, dim=2, descending=True)
    p_cumulative = torch.cumsum(p_sorted, dim=2)

    to_keep = p_cumulative <= top_p

    # Always keep at least one, so shift to the right
    to_keep[..., 1:] = to_keep[..., :-1].clone()
    to_keep[..., 0] = 1

    to_keep = sorted_indices[to_keep]

    p = p[..., to_keep]

    token_p, token_idx = sample(p, temperature)

    token_idx = to_keep[..., token_idx]

    return token_p, token_idx


def sample(p, temperature=1.0):
    """
    TODO : make it work for batch_size > 0

    :param p: shape (1, 1, num_tokens)
    :param temperature:
    :return:
    """
    p = torch.pow(p, 1.0 / temperature)
    p_sum = torch.sum(p, dim=2)
    p = p / p_sum[:, :, None]

    token_idx = torch.multinomial(p.view(-1), 1)
    token_p = torch.gather(p, 2, token_idx.view(1, -1, 1))

    token_idx = token_idx.view(1, -1)
    token_p = token_p.view(1, -1)

    return token_p, token_idx
