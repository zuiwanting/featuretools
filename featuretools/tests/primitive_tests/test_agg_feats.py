# -*- coding: utf-8 -*-

from datetime import datetime
from math import isnan

import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools.primitives import (  # NMostCommon,
    Count,
    Mean,
    Median,
    NumTrue,
    Sum,
    TimeSinceFirst,
    TimeSinceLast,
    get_aggregation_primitives
)
from featuretools.primitives.base import (
    AggregationPrimitive,
    make_agg_primitive
)
from featuretools.primitives.utils import (
    PrimitivesDeserializer,
    serialize_primitive
)
from featuretools.synthesis.deep_feature_synthesis import (
    DeepFeatureSynthesis,
    check_stacking,
    match
)
from featuretools.tests.testing_utils import feature_with_name
from featuretools.variable_types import (
    Datetime,
    DatetimeTimeIndex,
    Discrete,
    Index,
    Numeric,
    Variable
)


@pytest.fixture
def test_primitive():
    class TestAgg(AggregationPrimitive):
        name = "test"
        input_types = [Numeric]
        return_type = Numeric
        stack_on = []

        def get_function(self):
            return None

    return TestAgg


def test_get_depth(es):
    log_id_feat = es['log']['id']
    customer_id_feat = es['customers']['id']
    count_logs = ft.Feature(log_id_feat, parent_entity=es['sessions'], primitive=Count)
    sum_count_logs = ft.Feature(count_logs, parent_entity=es['customers'], primitive=Sum)
    num_logs_greater_than_5 = sum_count_logs > 5
    count_customers = ft.Feature(customer_id_feat,
                                 parent_entity=es[u'régions'],
                                 where=num_logs_greater_than_5,
                                 primitive=Count)
    num_customers_region = ft.Feature(count_customers, entity=es["customers"])

    depth = num_customers_region.get_depth()
    assert depth == 5


def test_makes_count(es):
    dfs = DeepFeatureSynthesis(target_entity_id='sessions',
                               entityset=es,
                               agg_primitives=[Count],
                               trans_primitives=[])

    features = dfs.build_features()
    assert feature_with_name(features, 'device_type')
    assert feature_with_name(features, 'customer_id')
    assert feature_with_name(features, u'customers.région_id')
    assert feature_with_name(features, 'customers.age')
    assert feature_with_name(features, 'COUNT(log)')
    assert feature_with_name(features, 'customers.COUNT(sessions)')
    assert feature_with_name(features, u'customers.régions.language')
    assert feature_with_name(features, 'customers.COUNT(log)')


def test_count_null_and_make_agg_primitive(es):
    def count_func(values, count_null=False):
        if len(values) == 0:
            return 0

        if count_null:
            values = values.fillna(0)

        return values.count()

    def count_generate_name(self, base_feature_names, relationship_path_name,
                            parent_entity_id, where_str, use_prev_str):
        return u"COUNT(%s%s%s)" % (relationship_path_name,
                                   where_str,
                                   use_prev_str)

    Count = make_agg_primitive(count_func, [[Index], [Variable]], Numeric,
                               name="count", stack_on_self=False,
                               cls_attributes={"generate_name": count_generate_name})
    count_null = ft.Feature(es['log']['value'], parent_entity=es['sessions'], primitive=Count(count_null=True))
    feature_matrix = ft.calculate_feature_matrix([count_null], entityset=es)
    values = [5, 4, 1, 2, 3, 2]
    assert (values == feature_matrix[count_null.get_name()]).all()


def test_check_input_types(es):
    count = ft.Feature(es["sessions"]["id"], parent_entity=es["customers"], primitive=Count)
    mean = ft.Feature(count, parent_entity=es[u"régions"], primitive=Mean)
    assert mean._check_input_types()

    boolean = count > 3
    mean = ft.Feature(count, parent_entity=es[u"régions"], where=boolean, primitive=Mean)
    assert mean._check_input_types()


def test_mean_nan(es):
    array = pd.Series([5, 5, 5, 5, 5])
    mean_func_nans_default = Mean().get_function()
    mean_func_nans_false = Mean(skipna=False).get_function()
    mean_func_nans_true = Mean(skipna=True).get_function()
    assert mean_func_nans_default(array) == 5
    assert mean_func_nans_false(array) == 5
    assert mean_func_nans_true(array) == 5
    array = pd.Series([5, np.nan, np.nan, np.nan, np.nan, 10])
    assert mean_func_nans_default(array) == 7.5
    assert isnan(mean_func_nans_false(array))
    assert mean_func_nans_true(array) == 7.5
    array_nans = pd.Series([np.nan, np.nan, np.nan, np.nan])
    assert isnan(mean_func_nans_default(array_nans))
    assert isnan(mean_func_nans_false(array_nans))
    assert isnan(mean_func_nans_true(array_nans))

    # test naming
    default_feat = ft.Feature(es["log"]["value"],
                              parent_entity=es["customers"],
                              primitive=Mean)
    assert default_feat.get_name() == "MEAN(log.value)"
    ignore_nan_feat = ft.Feature(es["log"]["value"],
                                 parent_entity=es["customers"],
                                 primitive=Mean(skipna=True))
    assert ignore_nan_feat.get_name() == "MEAN(log.value)"
    include_nan_feat = ft.Feature(es["log"]["value"],
                                  parent_entity=es["customers"],
                                  primitive=Mean(skipna=False))
    assert include_nan_feat.get_name() == "MEAN(log.value, skipna=False)"


def test_base_of_and_stack_on_heuristic(es, test_primitive):
    child = ft.Feature(es["sessions"]["id"], parent_entity=es["customers"], primitive=Count)
    test_primitive.stack_on = []
    child.primitive.base_of = []
    assert not check_stacking(test_primitive(), [child])

    test_primitive.stack_on = []
    child.primitive.base_of = None
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = []
    child.primitive.base_of = [test_primitive]
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = None
    child.primitive.base_of = []
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = None
    child.primitive.base_of = None
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = None
    child.primitive.base_of = [test_primitive]
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = [type(child.primitive)]
    child.primitive.base_of = []
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = [type(child.primitive)]
    child.primitive.base_of = None
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = [type(child.primitive)]
    child.primitive.base_of = [test_primitive]
    assert check_stacking(test_primitive(), [child])


def test_stack_on_self(es, test_primitive):
    # test stacks on self
    child = ft.Feature(es['log']['value'], parent_entity=es[u'régions'], primitive=test_primitive)
    test_primitive.stack_on = []
    child.primitive.base_of = []
    test_primitive.stack_on_self = False
    child.primitive.stack_on_self = False
    assert not check_stacking(test_primitive(), [child])

    test_primitive.stack_on_self = True
    assert check_stacking(test_primitive(), [child])

    test_primitive.stack_on = None
    test_primitive.stack_on_self = False
    assert not check_stacking(test_primitive(), [child])


def test_init_and_name(es):
    session = es['sessions']
    log = es['log']

    features = [ft.Feature(v) for v in log.variables]
    for agg_prim in get_aggregation_primitives().values():

        input_types = agg_prim.input_types
        if type(input_types[0]) != list:
            input_types = [input_types]

        # test each allowed input_types for this primitive
        for it in input_types:
            # use the input_types matching function from DFS
            matching_types = match(it, features)
            if len(matching_types) == 0:
                raise Exception("Agg Primitive %s not tested" % agg_prim.name)
            for t in matching_types:
                instance = ft.Feature(t, parent_entity=session, primitive=agg_prim)

                # try to get name and calculate
                instance.get_name()
                ft.calculate_feature_matrix([instance], entityset=es).head(5)


def test_invalid_init_args(diamond_es):
    transaction_relationships = diamond_es.get_forward_relationships('transactions')
    transaction_to_store = next(r for r in transaction_relationships
                                if r.parent_entity.id == 'stores')
    error_text = 'parent_entity must match first relationship in path'
    with pytest.raises(AssertionError, match=error_text):
        ft.AggregationFeature(diamond_es['transactions']['amount'],
                              diamond_es['customers'],
                              ft.primitives.Mean,
                              relationship_path=[transaction_to_store])

    store_to_region = diamond_es.get_forward_relationships('stores')[0]
    error_text = 'Base feature must be defined on the entity at the end of relationship_path'
    with pytest.raises(AssertionError, match=error_text):
        ft.AggregationFeature(diamond_es['transactions']['amount'],
                              diamond_es['regions'],
                              ft.primitives.Mean,
                              relationship_path=[store_to_region])


def test_init_with_multiple_possible_paths(diamond_es):
    error_text = "There are multiple possible paths to the base entity. " \
                 "You must specify a relationship path."
    with pytest.raises(RuntimeError, match=error_text):
        ft.AggregationFeature(diamond_es['transactions']['amount'],
                              diamond_es['regions'],
                              ft.primitives.Mean)

    transaction_relationships = diamond_es.get_forward_relationships('transactions')
    transaction_to_customer = next(r for r in transaction_relationships
                                   if r.parent_entity.id == 'customers')
    customer_to_region = diamond_es.get_forward_relationships('customers')[0]
    # Does not raise if path specified.
    ft.AggregationFeature(diamond_es['transactions']['amount'],
                          diamond_es['regions'],
                          ft.primitives.Mean,
                          relationship_path=[customer_to_region, transaction_to_customer])


def test_init_with_single_possible_path(diamond_es):
    # This uses diamond_es to test that there being a cycle somewhere in the
    # graph doesn't cause an error.
    feat = ft.AggregationFeature(diamond_es['transactions']['amount'],
                                 diamond_es['customers'],
                                 ft.primitives.Mean)
    transaction_relationships = diamond_es.get_forward_relationships('transactions')
    transaction_to_customer = next(r for r in transaction_relationships
                                   if r.parent_entity.id == 'customers')
    assert feat.relationship_path == [transaction_to_customer]


def test_init_with_no_path(diamond_es):
    error_text = 'No path from "transactions" to "customers" found.'
    with pytest.raises(RuntimeError, match=error_text):
        ft.AggregationFeature(diamond_es['customers']['name'],
                              diamond_es['transactions'],
                              ft.primitives.Count)

    error_text = 'No path from "transactions" to "transactions" found.'
    with pytest.raises(RuntimeError, match=error_text):
        ft.AggregationFeature(diamond_es['transactions']['amount'],
                              diamond_es['transactions'],
                              ft.primitives.Mean)


def test_name_with_multiple_possible_paths(diamond_es):
    transaction_relationships = diamond_es.get_forward_relationships('transactions')
    transaction_to_customer = next(r for r in transaction_relationships
                                   if r.parent_entity.id == 'customers')
    customer_to_region = diamond_es.get_forward_relationships('customers')[0]
    # Does not raise if path specified.
    feat = ft.AggregationFeature(diamond_es['transactions']['amount'],
                                 diamond_es['regions'],
                                 ft.primitives.Mean,
                                 relationship_path=[customer_to_region, transaction_to_customer])

    assert feat.get_name() == "MEAN(customers.transactions.amount)"


def test_copy(games_es):
    home_games = next(r for r in games_es.relationships
                      if r.child_variable.id == 'home_team_id')
    feat = ft.AggregationFeature(games_es['games']['home_team_score'],
                                 games_es['teams'],
                                 relationship_path=[home_games],
                                 primitive=ft.primitives.Mean)
    copied = feat.copy()
    assert copied.entity == feat.entity
    assert copied.base_features == feat.base_features
    assert copied.relationship_path == feat.relationship_path
    assert copied.primitive == feat.primitive


def test_serialization(es):
    primitives_deserializer = PrimitivesDeserializer()
    value = ft.IdentityFeature(es['log']['value'])
    primitive = ft.primitives.Max()
    max1 = ft.AggregationFeature(value, es['customers'], primitive)

    path = next(es.find_backward_paths('customers', 'log'))
    dictionary = {
        'base_features': [value.unique_name()],
        'relationship_path': [r.to_dictionary() for r in path],
        'primitive': serialize_primitive(primitive),
        'where': None,
        'use_previous': None,
    }

    assert dictionary == max1.get_arguments()
    assert max1 == \
        ft.AggregationFeature.from_dictionary(dictionary, es,
                                              {value.unique_name(): value},
                                              primitives_deserializer)

    is_purchased = ft.IdentityFeature(es['log']['purchased'])
    use_previous = ft.Timedelta(3, 'd')
    max2 = ft.AggregationFeature(value, es['customers'], primitive,
                                 where=is_purchased, use_previous=use_previous)

    dictionary = {
        'base_features': [value.unique_name()],
        'relationship_path': [r.to_dictionary() for r in path],
        'primitive': serialize_primitive(primitive),
        'where': is_purchased.unique_name(),
        'use_previous': use_previous.get_arguments(),
    }

    assert dictionary == max2.get_arguments()
    dependencies = {
        value.unique_name(): value,
        is_purchased.unique_name(): is_purchased
    }
    assert max2 == \
        ft.AggregationFeature.from_dictionary(dictionary, es, dependencies,
                                              primitives_deserializer)


def test_time_since_last(es):
    f = ft.Feature(es["log"]["datetime"], parent_entity=es["customers"], primitive=TimeSinceLast)
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [131376000.0, 131289534.0, 131287797.0]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)


def test_time_since_first(es):
    f = ft.Feature(es["log"]["datetime"], parent_entity=es["customers"], primitive=TimeSinceFirst)
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [131376600.0, 131289600.0, 131287800.0]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)


def test_median(es):
    f = ft.Feature(es["log"]["value_many_nans"], parent_entity=es["customers"], primitive=Median)
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [1, 3, np.nan]
    np.testing.assert_equal(fm[f.get_name()].values, correct)


def test_agg_same_method_name(es):
    """
        Pandas relies on the function name when calculating aggregations. This means if a two
        primitives with the same function name are applied to the same column, pandas
        can't differentiate them. We have a work around to this based on the name property
        that we test here.
    """

    # test with normally defined functions
    def custom_primitive(x):
        return x.sum()

    Sum = make_agg_primitive(custom_primitive, input_types=[Numeric],
                             return_type=Numeric, name="sum")

    def custom_primitive(x):
        return x.max()

    Max = make_agg_primitive(custom_primitive, input_types=[Numeric],
                             return_type=Numeric, name="max")

    f_sum = ft.Feature(es["log"]["value"], parent_entity=es["customers"], primitive=Sum)
    f_max = ft.Feature(es["log"]["value"], parent_entity=es["customers"], primitive=Max)

    fm = ft.calculate_feature_matrix([f_sum, f_max], entityset=es)
    assert fm.columns.tolist() == [f_sum.get_name(), f_max.get_name()]

    # test with lambdas
    Sum = make_agg_primitive(lambda x: x.sum(), input_types=[Numeric],
                             return_type=Numeric, name="sum")
    Max = make_agg_primitive(lambda x: x.max(), input_types=[Numeric],
                             return_type=Numeric, name="max")

    f_sum = ft.Feature(es["log"]["value"], parent_entity=es["customers"], primitive=Sum)
    f_max = ft.Feature(es["log"]["value"], parent_entity=es["customers"], primitive=Max)
    fm = ft.calculate_feature_matrix([f_sum, f_max], entityset=es)
    assert fm.columns.tolist() == [f_sum.get_name(), f_max.get_name()]


def test_time_since_last_custom(es):
    def time_since_last(values, time=None):
        time_since = time - values.iloc[0]
        return time_since.total_seconds()

    TimeSinceLast = make_agg_primitive(time_since_last,
                                       [DatetimeTimeIndex],
                                       Numeric,
                                       name="time_since_last",
                                       uses_calc_time=True)
    f = ft.Feature(es["log"]["datetime"], parent_entity=es["customers"], primitive=TimeSinceLast)
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [131376600, 131289600, 131287800]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        TimeSinceLast = make_agg_primitive(time_since_last,
                                           [DatetimeTimeIndex],
                                           Numeric,
                                           uses_calc_time=False)


def test_custom_primitive_time_as_arg(es):
    def time_since_last(values, time):
        time_since = time - values.iloc[0]
        return time_since.total_seconds()

    TimeSinceLast = make_agg_primitive(time_since_last,
                                       [DatetimeTimeIndex],
                                       Numeric,
                                       uses_calc_time=True)
    assert TimeSinceLast.name == "time_since_last"
    f = ft.Feature(es["log"]["datetime"], parent_entity=es["customers"], primitive=TimeSinceLast)
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [131376600, 131289600, 131287800]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_agg_primitive(time_since_last,
                           [DatetimeTimeIndex],
                           Numeric,
                           uses_calc_time=False)


def test_custom_primitive_multiple_inputs(es):
    def mean_sunday(numeric, datetime):
        '''
        Finds the mean of non-null values of a feature that occurred on Sundays
        '''
        days = pd.DatetimeIndex(datetime).weekday.values
        df = pd.DataFrame({'numeric': numeric, 'time': days})
        return df[df['time'] == 6]['numeric'].mean()

    MeanSunday = make_agg_primitive(function=mean_sunday,
                                    input_types=[Numeric, Datetime],
                                    return_type=Numeric)

    fm, features = ft.dfs(entityset=es,
                          target_entity="sessions",
                          agg_primitives=[MeanSunday],
                          trans_primitives=[])
    mean_sunday_value = pd.Series([None, None, None, 2.5, 7, None])
    iterator = zip(fm["MEAN_SUNDAY(log.value, datetime)"], mean_sunday_value)
    for x, y in iterator:
        assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

    es.add_interesting_values()
    mean_sunday_value_priority_0 = pd.Series([None, None, None, 2.5, 0, None])
    fm, features = ft.dfs(entityset=es,
                          target_entity="sessions",
                          agg_primitives=[MeanSunday],
                          trans_primitives=[],
                          where_primitives=[MeanSunday])
    where_feat = "MEAN_SUNDAY(log.value, datetime WHERE priority_level = 0)"
    for x, y in zip(fm[where_feat], mean_sunday_value_priority_0):
        assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))


def test_custom_primitive_default_kwargs(es):
    def sum_n_times(numeric, n=1):
        return np.nan_to_num(numeric).sum(dtype=np.float) * n

    SumNTimes = make_agg_primitive(function=sum_n_times,
                                   input_types=[Numeric],
                                   return_type=Numeric)

    sum_n_1_n = 1
    sum_n_1_base_f = ft.Feature(es['log']['value'])
    sum_n_1 = ft.Feature([sum_n_1_base_f], parent_entity=es['sessions'], primitive=SumNTimes(n=sum_n_1_n))
    sum_n_2_n = 2
    sum_n_2_base_f = ft.Feature(es['log']['value_2'])
    sum_n_2 = ft.Feature([sum_n_2_base_f], parent_entity=es['sessions'], primitive=SumNTimes(n=sum_n_2_n))
    assert sum_n_1_base_f == sum_n_1.base_features[0]
    assert sum_n_1_n == sum_n_1.primitive.kwargs['n']
    assert sum_n_2_base_f == sum_n_2.base_features[0]
    assert sum_n_2_n == sum_n_2.primitive.kwargs['n']


def test_makes_numtrue(es):
    dfs = DeepFeatureSynthesis(target_entity_id='sessions',
                               entityset=es,
                               agg_primitives=[NumTrue],
                               trans_primitives=[])
    features = dfs.build_features()
    assert feature_with_name(features, 'customers.NUM_TRUE(log.purchased)')
    assert feature_with_name(features, 'NUM_TRUE(log.purchased)')


def test_make_three_most_common(es):
    def pd_top3(x):
        array = np.array(x.value_counts()[:3].index)
        if len(array) < 3:
            filler = np.full(3 - len(array), np.nan)
            array = np.append(array, filler)
        return array

    NMostCommoner = make_agg_primitive(function=pd_top3,
                                       input_types=[Discrete],
                                       return_type=Discrete,
                                       number_output_features=3)

    fm, features = ft.dfs(entityset=es,
                          target_entity="customers",
                          agg_primitives=[NMostCommoner],
                          trans_primitives=[])

    true_results = pd.DataFrame([
        ['coke zero', 'toothpaste', "car"],
        ['coke zero', 'Haribo sugar-free gummy bears', np.nan],
        ['taco clock', np.nan, np.nan]
    ])
    df = fm[["PD_TOP3(log.product_id)__%s" % i for i in range(3)]]
    for i in range(df.shape[0]):
        if i == 0:
            # coke zero and toothpaste have same number of occurrences
            # so just check that the top two match
            assert set(true_results.iloc[i].values[:2]) == set(df.iloc[i].values[:2])
            assert df.iloc[0].values[2] in ("brown bag", "car")
        else:
            for i1, i2 in zip(true_results.iloc[i], df.iloc[i]):
                assert (pd.isnull(i1) and pd.isnull(i2)) or (i1 == i2)
