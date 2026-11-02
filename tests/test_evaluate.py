from evaluate import lead3


class TestLead3:
    def test_takes_the_first_three_sentences(self):
        assert lead3("One. Two. Three. Four.") == "One. Two. Three."

    def test_returns_everything_when_there_are_fewer_than_three(self):
        assert lead3("One. Two.") == "One. Two."

    def test_ignores_empty_segments_from_consecutive_periods(self):
        assert lead3("One.. Two. Three. Four.") == "One. Two. Three."

    def test_empty_text_gives_an_empty_summary(self):
        assert lead3("") == ""

    def test_text_without_a_terminal_period_still_scores(self):
        assert lead3("Only one sentence") == "Only one sentence."
