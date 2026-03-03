# tests/test_stats_utils.py

import pytest
from analysis.stats_utils import mann_kendall_test, normalize_yearly_counts, calculate_growth_rate


class TestMannKendall:

    def test_increasing_trend(self):
        yearly = {2017: 10, 2018: 30, 2019: 80, 2020: 200, 2021: 400}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'increasing'
        assert result['significant'] == True
        assert result['p_value'] < 0.05

    def test_decreasing_trend(self):
        yearly = {2017: 400, 2018: 300, 2019: 200, 2020: 100, 2021: 50}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'decreasing'
        assert result['significant'] == True

    def test_insufficient_data(self):
        yearly = {2020: 100, 2021: 110}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'no trend'
        assert result['significant'] == False

    def test_empty_data(self):
        yearly = {}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'no trend'


class TestNormalizeYearly:

    def test_normalize(self):
        yearly = {2020: 100, 2021: 200, 2022: 300}
        normalized = normalize_yearly_counts(yearly)
        assert abs(sum(normalized.values()) - 1.0) < 0.001
        # 100/600 = 0.166666...
        assert abs(normalized[2020] - 0.1666) < 0.001


class TestGrowthRate:

    def test_compound_growth(self):
        yearly = {2020: 100, 2021: 150, 2022: 225}  # 50% growth each year
        rate = calculate_growth_rate(yearly, method='compound')
        assert abs(rate - 0.50) < 0.01

    def test_zero_growth(self):
        yearly = {2020: 100, 2021: 100, 2022: 100}
        rate = calculate_growth_rate(yearly)
        assert abs(rate) < 0.01
