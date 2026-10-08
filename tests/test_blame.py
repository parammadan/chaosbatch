from chaosbatch.guard import BlameBook


def test_poison_row_is_quarantined_after_two_solo_deaths():
    b = BlameBook()
    b.report_death("n1", list(range(50)))           # batch dies: 50 suspects
    assert len(b.state()["suspects"]) == 50 and not b.quarantined
    b.report_death("n1", [17])                       # 17 alone in flight, dies
    assert 17 not in b.quarantined                    # one solo death isn't proof
    b.report_death("n2", [17])
    assert b.quarantined == {17}
    assert 17 not in b.state()["suspects"]


def test_innocent_suspects_are_never_quarantined_by_batch_deaths():
    b = BlameBook()
    for _ in range(5):
        b.report_death("n1", [1, 2, 3])               # always in a crowd
    assert not b.quarantined


def test_device_is_fenced_after_two_canary_deaths_once():
    b = BlameBook()
    assert b.report_canary_death("bad") is False
    assert b.report_canary_death("bad") is True       # crosses threshold
    assert b.report_canary_death("bad") is False      # fence only created once
    assert b.fenced == {"bad"}


def test_canary_deaths_never_blame_rows():
    b = BlameBook()
    b.report_canary_death("bad")
    b.report_canary_death("bad")
    assert not b.suspicion and not b.quarantined
