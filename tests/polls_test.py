from src.polls import *

def test_get_weighted_poll_avg():
    url = url_19
    col_dict = col_dict19
    poll_avg = get_weighted_poll_avg(url, col_dict)
    assert len(poll_avg) == 7, "Poll average should contain 7 parties."
    for party in col_dict.values():
        assert party in poll_avg.index, f"{party} should be in the poll average index."